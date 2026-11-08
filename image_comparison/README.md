# Image Generation Comparison

Streamlit application for comparing paired PSPStain images from six columns:

1. Input (`MIST/val_pretrained/images/real_A`)
2. Ground Truth (`MIST/val_pretrained/images/real_B`)
3. MIST (`MIST/val_pretrained/images/fake_B`)
4. Hibou 15 (`hibou_15_patches/val_latest/images/fake_B`)
5. Hibou 25 (`hibou_25_patches/val_latest/images/fake_B`)
6. Random 15 (`random_15/val_latest/images/fake_B`)

## Expected directory layout

```text
PSPStain/
├── results/
│   ├── MIST/val_pretrained/images/{real_A,real_B,fake_B}/
│   ├── MIST/val_pretrained/comb/fake_B/
│   ├── MIST/val_pretrained/good/fake_B/
│   ├── MIST/val_pretrained/norm/fake_B/
│   ├── hibou_15_patches/val_latest/images/fake_B/
│   ├── hibou_25_patches/val_latest/images/fake_B/
│   └── random_15/val_latest/images/fake_B/
└── image_comparison/
    ├── app.py
    ├── requirements.txt
    └── run.sh
```

## Install

From `PSPStain/image_comparison`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your cluster requires a particular Python module or environment, use that environment instead of creating a venv.

## Run

Run on an allocated compute node:

```bash
cd /path/to/PSPStain/image_comparison
source .venv/bin/activate
bash run.sh
```

Optional environment variables:

```bash
PORT=8501
ADDRESS=0.0.0.0
RESULTS_DIR=/absolute/path/to/PSPStain/results
```

The app defaults to `../results` relative to `app.py`.

## Features

- All examples, Comb, Good, Norm, and Other filters.
- 5, 10, or 20 examples per page.
- Pairing by exact filename.
- Runtime checks for missing/extra files.
- In-memory PNG previews; original files are never overwritten.
- Open Original dialog for full-resolution inspection.

## Network access

The app listening on a compute node does not automatically make it reachable from your laptop or colleagues. Use the access method approved/configured for Zhores (JupyterHub proxy or SSH port forwarding). The correct host, port-forwarding command, and whether colleagues can share the same URL depend on the cluster's network and JupyterHub configuration. Do not expose the port publicly unless the cluster administrators permit it.
