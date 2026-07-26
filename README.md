# CUT-4DFlow

Reference implementation for the paper **"ℂUT-4DFlow for Accelerated 4D-Flow MRI Reconstruction."**

## Installation

```bash
conda env create -f environment.yml
conda activate cut4dflow
```

## Data

The CMRxRecon 4D-flow dataset is available at
https://www.synapse.org/Synapse:syn64545434/wiki/630587 .
Download it and extract under `Data/` (see [`Data/README.md`](Data/README.md) for the
expected layout).

## Training

```bash
./scripts/train.sh configs/train.yaml
```

Edit `configs/train.yaml` for architecture and optimisation settings, or override the
data root with `CMRX_TRAIN_ROOT`. Checkpoints are written to `checkpoints/`.

## Reconstruction

```bash
CKPT=checkpoints/full_super_v2/best.ckpt ./scripts/recon.sh
```

This reconstructs a ValidationSet and writes the submission layout (sparse `.npz`) plus
optional per-case animations (`ANIM=1`). The model architecture is read from the checkpoint.

## Repository layout

```
src/
├── models/       complex layers, attention, denoiser, DC/WA, cascade, conditioning
├── data/         dataset, kt-Gaussian sampling, SSDU split, orientation, HDF5 reader
├── training/     training loop and losses
├── recon/        full-cycle reconstruction utilities
├── postprocess/  submission writer
├── utils/        FFT ops and evaluation metrics
└── baselines/    compressed-sensing (locally low-rank) baseline
configs/          training configuration
scripts/          conda entry points (train / recon)
Data/             dataset (downloaded separately)
checkpoints/      trained weights
```

## Acknowledgements

This work was supervised by:

- [Luigi Perotti](https://github.com/luigiemp)
- [Kevin Moulin](https://github.com/KMoulin)
- [Dennis Ogiermann](https://github.com/termi-official)
