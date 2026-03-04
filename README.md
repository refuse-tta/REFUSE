# REFUSE: A Normalization-free Paradigm with Entropy-regularized Feature Recalibration for Test-Time Adaptation

> **Anonymous submission to ECCV 2026**

## Overview

This repository provides the official implementation of **REFUSE**, along with baseline reproductions (TENT, EATA, SAR, BN-Adapt, Buffer) and interpretability analysis (CKA similarity, drift analysis).

## Repository Structure

```
REFUSE/
├── configs/default.yaml        # All hyper-parameters in one place
├── models/
│   └── resnet26.py             # ResNet-26 backbone (CIFAR-10 variant)
├── methods/
│   ├── refuse.py               # REFUSE (ours)
│   ├── tent.py                 # Fully-TTA: TENT
│   ├── eata.py                 # EATA
│   ├── sar.py                  # SAR
│   ├── bn_adapt.py             # BN-Adapt
│   └── buffer.py               # Buffer model
├── data/
│   └── corruptions.py          # CIFAR-10 & CIFAR-10-C loaders
├── analysis/
│   ├── cka.py                  # CKA similarity analysis
│   └── drift.py                # Feature drift analysis
├── scripts/
│   ├── train.py                # Train source ResNet-26
│   ├── adapt_models.py         # Adapt 4 models with different recipes
│   ├── run_refuse.py           # Evaluate REFUSE on CIFAR-10-C
│   ├── run_baselines.py        # Evaluate all baselines
│   └── run_analysis.py         # CKA + drift interpretability
├── checkpoints/                # Pre-trained & adapted weights (not tracked)
├── results/                    # Output tables and figures
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Data

1. **CIFAR-10** is auto-downloaded by `torchvision`.
2. **CIFAR-10-C**: Download from [Zenodo](https://zenodo.org/record/2535967) and place under `data/CIFAR-10-C/`.

## Usage

### 1. Train source model

```bash
python scripts/train.py --epochs 200 --save_path checkpoints/resnet26_cifar10.pth
```

### 2. Run REFUSE (ours)

```bash
python scripts/run_refuse.py \
    --checkpoint checkpoints/resnet26_cifar10.pth \
    --data_dir data/CIFAR-10-C \
    --severity 5
```

### 3. Run baselines

```bash
python scripts/run_baselines.py \
    --checkpoint checkpoints/resnet26_cifar10.pth \
    --data_dir data/CIFAR-10-C \
    --methods tent eata sar bn_adapt buffer \
    --severity 5
```

### 4. Adapt models & run analysis

```bash
python scripts/adapt_models.py --checkpoint checkpoints/resnet26_cifar10.pth
python scripts/run_analysis.py --adapted_dir checkpoints/
```

## Results

Results on CIFAR-10-C (severity 5), ResNet-26:

| Method    | Mean Accuracy (%) |
|-----------|----------------|
| Source    | 56.87              |
| BN-Adapt  | 76.85              |
| TENT      | 83.17              |
| EATA      | 82.16              |
| SAR       | 78.02              |
| Buffer    | 79.14              |
| **REFUSE (Ours)** | **76.86**  |

*(Fill in after reproducing.)*

## Citation

```
Anonymous ECCV 2026 submission – citation withheld during review.
```

## License

This project is released under the MIT License.
