import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """Load one split from a multi-view MAT file."""

    def __init__(self, root, split="train", scalers=None):
        path = Path(root)
        if path.suffix != ".mat":
            path = path.with_suffix(".mat")
        content = loadmat(path)

        view_ids = sorted(
            int(match.group(1))
            for key in content
            if (match := re.fullmatch(rf"x(\d+)_{re.escape(split)}", key))
        )
        if not view_ids:
            raise KeyError(f"No '{split}' views found in {path}")

        raw_views = [np.asarray(content[f"x{view_id}_{split}"]) for view_id in view_ids]
        if scalers is None:
            if split != "train":
                raise ValueError("Validation and test splits require train-fitted scalers.")
            scalers = [MinMaxScaler().fit(view) for view in raw_views]
        if len(scalers) != len(raw_views):
            raise ValueError(f"Expected {len(raw_views)} scalers, received {len(scalers)}")

        self.scalers = scalers
        self.views = {
            index: scaler.transform(view).astype(np.float32)
            for index, (scaler, view) in enumerate(zip(scalers, raw_views))
        }
        labels = np.asarray(content[f"gt_{split}"]).reshape(-1)
        if labels.size and labels.min() == 1:
            labels = labels - 1
        self.labels = labels.astype(np.int64)

    def __getitem__(self, index):
        views = {view: values[index] for view, values in self.views.items()}
        return views, self.labels[index], index

    def __len__(self):
        return len(self.labels)

