import argparse
from pathlib import Path

import torch

from data import MultiViewDataset
from model import MultiViewClassifier
from train import evaluate, make_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a frozen checkpoint")
    parser.add_argument("--dataset", default="pacs_art4V")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.output_dir / f"{args.dataset}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    data_path = args.data_dir / args.dataset
    source = MultiViewDataset(data_path, "train")
    target = MultiViewDataset(data_path, "test", scalers=source.scalers)
    loader = make_loader(
        target, args.batch_size, False, seed=0, num_workers=args.num_workers
    )

    model = MultiViewClassifier(
        checkpoint["classes"], checkpoint["input_dims"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    feature_weights = checkpoint["feature_weights"].to(device)
    loss, accuracy, auc = evaluate(
        model, loader, checkpoint["classes"], device, feature_weights
    )
    print(
        f"{args.dataset}: loss={loss:.4f}, accuracy={accuracy:.6f}, "
        f"AUROC={auc:.6f}"
    )


if __name__ == "__main__":
    main()

