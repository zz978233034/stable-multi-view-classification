# MvSC

A compact source-only implementation of multi-view classification with adaptive
sample and representation-dimension weighting (ASDW).

The training code never loads the target/test split. Checkpoint selection uses
only source-domain validation data, and final target evaluation is a separate
command.

## Files

- `train.py`: training and source-validation checkpoint selection.
- `evaluate.py`: evaluation of a frozen checkpoint on the target split.
- `model.py`: multi-view encoders and fusion classifier.
- `weighting.py`: ASDW optimization with one fixed configuration.
- `data.py`: MAT dataset loading and train-fitted normalization.
- `prepare_pacs.py`: optional PACS four-view feature extraction.

Datasets, pretrained weights, checkpoints, and experiment outputs are not
included in the repository.

## Installation

```bash
pip install -r requirements.txt
```

## Data format

Place datasets in `data/`. Each `<dataset>.mat` file should contain:

```text
x1_train, x2_train, ..., gt_train
x1_val,   x2_val,   ..., gt_val      # optional
x1_test,  x2_test,  ..., gt_test
```

If no validation split is present, `train.py` creates a fixed stratified split
from the source training data.

To build the four PACS tasks from the official images and split lists:

```bash
python prepare_pacs.py --data-root /path/to/PACS
```

This creates `data/pacs_art4V.mat`, `data/pacs_cartoon4V.mat`,
`data/pacs_photo4V.mat`, and `data/pacs_sketch4V.mat`.

## Training and evaluation

```bash
python train.py --dataset pacs_art4V
python evaluate.py --dataset pacs_art4V
```

Only common runtime options are exposed (`--epochs`, `--batch-size`, `--lr`,
`--seed`, and paths). Method settings are fixed near the top of `weighting.py`
instead of being duplicated as unrelated command-line switches.

