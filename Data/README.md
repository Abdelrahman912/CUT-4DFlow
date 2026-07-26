# Data

CUT-4DFlow is trained and evaluated on the **CMRxRecon 4D-Flow** dataset.

**Download:** https://www.synapse.org/Synapse:syn64545434/wiki/630587

After downloading and extracting, place the data under this `Data/` folder so the
default paths in `configs/train.yaml` and `scripts/recon.sh` resolve, e.g.:

```
Data/
├── TaskR1R2/
│   ├── TrainSet/Aorta/<Center>/<Vendor>/<Patient>/{kdata_full.mat, coilmap.mat, segmask.mat, params.csv}
│   └── ValidationSet/Aorta/<Center>/<Vendor>/<Patient>/{kdata_ktGaussian{R}.mat, coilmap.mat, segmask.mat, params.csv}
├── TaskS1/ValidationSet/Aorta/...
└── TaskS2/ValidationSet/{Carotid,Cerebrovascular,PortalVein,RenalArtery}/...
```

Each patient volume provides multi-coil k-space, coil-sensitivity maps, a segmentation
mask, and an acquisition-parameter CSV (VENC, spatial order, field strength, …).

Override the default location at run time with `CMRX_TRAIN_ROOT` (training) or
`VAL_ROOT` (reconstruction).
