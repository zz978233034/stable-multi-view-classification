import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data import MultiViewDataset
from model import EMBED_DIM, MultiViewClassifier, weighted_cross_entropy
from weighting import make_rff_parameters, optimize_weights


WEIGHT_DECAY = 1e-5
CLASS_BALANCE_POWER = 0.3
VALIDATION_RATIO = 0.1

def print_flops(model, input_dims, device):
    inputs = {i: torch.randn(1, dim, device=device) for i, dim in enumerate(input_dims)}
    model.eval()
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
    ) as profiler:
        model(inputs, torch.ones(100, device=device))

    flops = sum(item.flops for item in profiler.key_averages())
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"FLOPs: {flops / 1e6:.3f} M | Params: {params / 1e6:.3f} M")
def parse_args():
    parser = argparse.ArgumentParser(description="Source-only multi-view training")
    parser.add_argument("--dataset", default="pacs_art4V")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch, device):
    views, targets, indices = batch
    views = {key: value.float().to(device) for key, value in views.items()}
    return views, targets.long().to(device), indices.long().to(device)


def class_weights(labels, classes, device):
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    weights = np.power(counts.sum() / np.maximum(counts, 1), CLASS_BALANCE_POWER)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(
    model,
    loader,
    optimizer,
    sample_weights,
    feature_weights,
    label_weights,
    device,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in loader:
        views, targets, indices = move_batch(batch, device)
        batch_weights = sample_weights[indices].detach()
        optimizer.zero_grad()
        view_logits, fused_logits, _ = model(views, feature_weights.detach())
        loss = sum(
            weighted_cross_entropy(logits, targets, batch_weights, label_weights)
            for logits in view_logits
        )
        loss += weighted_cross_entropy(
            fused_logits, targets, batch_weights, label_weights
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        total_correct += (fused_logits.argmax(1) == targets).sum().item()
        total_samples += len(targets)
    return total_loss / total_samples, total_correct / total_samples


def collect_representations(model, loader, device):
    model.eval()
    features = []
    indices = []
    with torch.no_grad():
        for batch in loader:
            views, _, batch_indices = move_batch(batch, device)
            _, _, fused = model(views)
            features.append(fused)
            indices.append(batch_indices)
    return torch.cat(features), torch.cat(indices)


def evaluate(model, loader, classes, device, feature_weights):
    model.eval()
    logits = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            views, batch_targets, _ = move_batch(batch, device)
            _, fused_logits, _ = model(views, feature_weights)
            logits.append(fused_logits)
            targets.append(batch_targets)
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    loss = F.cross_entropy(logits, targets).item()
    accuracy = (logits.argmax(1) == targets).float().mean().item()
    probabilities = F.softmax(logits, dim=1).cpu().numpy()
    one_hot = np.eye(classes, dtype=np.float32)[targets.cpu().numpy()]
    try:
        auc = roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return loss, accuracy, auc


def load_source_splits(data_path, seed):
    source = MultiViewDataset(data_path, "train")
    try:
        validation = MultiViewDataset(data_path, "val", scalers=source.scalers)
        training = source
        protocol = "provided source validation split"
    except KeyError:
        train_indices, val_indices = train_test_split(
            np.arange(len(source)),
            test_size=VALIDATION_RATIO,
            random_state=seed,
            stratify=source.labels,
        )
        training = Subset(source, train_indices.tolist())
        validation = Subset(source, val_indices.tolist())
        protocol = "fixed stratified source validation split"
    return source, training, validation, protocol


def save_history(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_dir / args.dataset
    source, training, validation, protocol = load_source_splits(data_path, args.seed)

    input_dims = [source.views[index].shape[1] for index in range(len(source.views))]
    classes = int(source.labels.max() + 1)
    model = MultiViewClassifier(classes, input_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    train_loader = make_loader(
        training, args.batch_size, True, args.seed, args.num_workers
    )
    train_eval_loader = make_loader(
        training, args.batch_size, False, args.seed, args.num_workers
    )
    val_loader = make_loader(
        validation, args.batch_size, False, args.seed, args.num_workers
    )
    indices = np.asarray(getattr(training, "indices", np.arange(len(source))))
    label_weights = class_weights(source.labels[indices], classes, device)
    sample_weights = torch.ones(len(source), device=device)
    feature_weights = torch.ones(EMBED_DIM, device=device)
    rff = make_rff_parameters(device, torch.float32, args.seed + 10_000)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.dataset}.pth"
    history_path = args.output_dir / f"{args.dataset}_history.csv"
    best_accuracy = -1.0
    best_auc = -float("inf")
    history = []

    print(
        f"Dataset={args.dataset} train={len(training)} val={len(validation)} "
        f"views={len(input_dims)} device={device}"
    )
    print(f"Protocol: {protocol}; target/test data are not loaded during training.")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, online_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            sample_weights,
            feature_weights,
            label_weights,
            device,
        )
        representations, representation_indices = collect_representations(
            model, train_eval_loader, device
        )
        updated_samples, feature_weights, stats = optimize_weights(
            representations,
            sample_weights[representation_indices],
            feature_weights,
            rff,
            args.seed + 20_000,
            epoch - 1,
        )
        sample_weights[representation_indices] = updated_samples

        _, train_accuracy, _ = evaluate(
            model, train_eval_loader, classes, device, feature_weights
        )
        val_loss, val_accuracy, val_auc = evaluate(
            model, val_loader, classes, device, feature_weights
        )
        scheduler.step()

        auc_score = val_auc if np.isfinite(val_auc) else -float("inf")
        if val_accuracy > best_accuracy or (
            np.isclose(val_accuracy, best_accuracy) and auc_score > best_auc
        ):
            best_accuracy = val_accuracy
            best_auc = auc_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "feature_weights": feature_weights.cpu(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "val_auc": val_auc,
                    "classes": classes,
                    "input_dims": input_dims,
                },
                checkpoint_path,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "online_train_accuracy": online_accuracy,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_auc": val_auc,
                **stats,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} train={train_accuracy:.4f} "
            f"val={val_accuracy:.4f} auc={val_auc:.4f} "
            f"dep={stats['dependency_before']:.3e}->{stats['dependency_after']:.3e} "
            f"updates={stats['sample_updates']}/{stats['feature_updates']} "
            f"time={time.time() - start:.1f}s"
        )

    save_history(history_path, history)
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

