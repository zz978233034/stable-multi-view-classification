"""Create four-view PACS MAT files from the official image dataset."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import savemat
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DOMAINS = ("art_painting", "cartoon", "photo", "sketch")
SHORT_NAMES = {"art_painting": "art", "cartoon": "cartoon", "photo": "photo", "sketch": "sketch"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract four PACS feature views")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Directory containing official split lists (default: data-root parent)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def read_list(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            relative_path, label = line.rsplit(maxsplit=1)
            records.append((relative_path.replace("\\", "/"), int(label) - 1))
    return records


def find_split(split_root, domain, split):
    filename = f"{domain}_{split}.txt"
    matches = sorted(split_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {split_root}")
    records = read_list(matches[0])
    for path in matches[1:]:
        if read_list(path) != records:
            raise ValueError(f"Conflicting split lists: {matches[0]} and {path}")
    return records


def build_tasks(data_root, split_root):
    class_names = sorted(
        path.name for path in (data_root / DOMAINS[0]).iterdir() if path.is_dir()
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    all_images = {}
    for domain in DOMAINS:
        all_images[domain] = [
            (path.relative_to(data_root).as_posix(), class_to_index[class_name])
            for class_name in class_names
            for path in sorted((data_root / domain / class_name).iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

    official_train = {
        domain: find_split(split_root, domain, "train") for domain in DOMAINS
    }
    official_val = {
        domain: find_split(split_root, domain, "val") for domain in DOMAINS
    }
    tasks = {}
    for target in DOMAINS:
        train, validation = [], []
        for domain_id, domain in enumerate(DOMAINS):
            if domain != target:
                train.extend((path, label, domain_id) for path, label in official_train[domain])
                validation.extend((path, label, domain_id) for path, label in official_val[domain])
        target_id = DOMAINS.index(target)
        test = [(path, label, target_id) for path, label in all_images[target]]
        tasks[target] = {"train": train, "val": validation, "test": test}

    for target, splits in tasks.items():
        for split, records in splits.items():
            missing = [path for path, _, _ in records if not (data_root / path).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{target}/{split}: {len(missing)} images are missing; first={missing[0]}"
                )
    return class_names, tasks


class ImageDataset(Dataset):
    def __init__(self, data_root, paths, transform):
        self.data_root = data_root
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.data_root / self.paths[index]).convert("RGB")
        return self.transform(image)


def feature_extractors(device):
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg = nn.Sequential(vgg.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    efficientnet = nn.Sequential(
        efficientnet.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    )
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()
    return [model.eval().to(device) for model in (vgg, resnet, efficientnet, vit)]


def extract_features(data_root, tasks, batch_size, num_workers, device):
    paths = sorted(
        {path for task in tasks.values() for split in task.values() for path, _, _ in split}
    )
    path_to_index = {path: index for index, path in enumerate(paths)}
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    loader = DataLoader(
        ImageDataset(data_root, paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    extractors = feature_extractors(device)
    chunks = [[] for _ in extractors]
    with torch.inference_mode():
        for images in tqdm(loader, desc="Extracting PACS features"):
            images = images.to(device, non_blocking=True)
            for view, extractor in enumerate(extractors):
                chunks[view].append(extractor(images).cpu().numpy())
    return path_to_index, [np.concatenate(part).astype(np.float32) for part in chunks]


def save_tasks(output_dir, class_names, tasks, path_to_index, features):
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, splits in tasks.items():
        payload = {
            "class_names": np.asarray(class_names, dtype=object),
            "domain_names": np.asarray(DOMAINS, dtype=object),
        }
        for split, records in splits.items():
            indices = np.asarray([path_to_index[path] for path, _, _ in records])
            for view, values in enumerate(features, start=1):
                payload[f"x{view}_{split}"] = values[indices]
            payload[f"gt_{split}"] = np.asarray([label for _, label, _ in records])
            payload[f"domain_{split}"] = np.asarray([domain for _, _, domain in records])
        destination = output_dir / f"pacs_{SHORT_NAMES[target]}4V.mat"
        savemat(destination, payload, do_compression=True)
        print(f"Saved {destination}")


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    split_root = (args.split_root or data_root.parent).resolve()
    class_names, tasks = build_tasks(data_root, split_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_index, features = extract_features(
        data_root, tasks, args.batch_size, args.num_workers, device
    )
    save_tasks(args.output_dir, class_names, tasks, path_to_index, features)


if __name__ == "__main__":
    main()

