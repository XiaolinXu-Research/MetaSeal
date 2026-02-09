# MetaSeal

Official implementation of **MetaSeal: Defending Against Image Attribution Forgery Through Content-Dependent Cryptographic Watermarks** (TMLR 2026).

MetaSeal is a watermarking framework for image attribution that combines semantic binding, cryptographic signatures, and robust secret recovery.

## Highlights

- Content-dependent attribution signals
- Cryptographic verification (public/private key workflow)
- Robust extraction under common benign transformations
- End-to-end demo notebook for caption -> sign -> encode -> recover -> verify



## Project Structure

```text
MetaSeal/
├── data/                # Input assets and demo data
├── images/              # Outputs and intermediate visualization files
├── model/               # Model artifacts / checkpoints
├── scripts/             # Training/inference/data-loading scripts
├── demo.ipynb           # End-to-end walkthrough
├── private_key.pem      # Private key of digital signature
├── public_key.pem       # Public key of digital signature
└── README.md
```
Feel free to generate your own key pair.

## Setup

### 1. Clone

```bash
git clone <YOUR_REPO_URL>
cd MetaSeal
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages used by the notebook and scripts (for example: `torch`, `transformers`, `cryptography`, `qrcode`, `pyzbar`, `Pillow`, `matplotlib`).

If you maintain a dependency file, add instructions here:

```bash
# Example
# pip install -r requirements.txt
```

## Quick Start

### Run the demo notebook

```bash
jupyter notebook demo.ipynb
```

The notebook demonstrates:

1. Semantic caption extraction from an input image
2. ECDSA signing of the caption
3. JSON payload + QR pattern generation
4. MetaSeal inference and secret recovery
5. Signature verification with the public key

## Training

Run training from the project root:

```bash
python scripts/train.py
```

Before training, update dataset paths and training hyperparameters in `scripts/config.py`:

- `TRAIN_COVER_PATH`, `TRAIN_SECRET_PATH`
- `VAL_COVER_PATH`, `VAL_SECRET_PATH`
- `batch_size`, `epochs`, `lr`, `device_ids`
- loss weights: `lamda_reconstruction`, `lamda_guide`, `lamda_low_frequency`

### Using Multiple Noise Layers

The implementation in `scripts/train.py` currently applies a JPEG simulation noise layer during training.
You can stack multiple noise layers (for example, JPEG + blur + resize) to improve robustness under more transformations.

Note: stronger/more noise layers usually improve robustness but can reduce visual quality (invisibility), so balancing these objectives may require carefully retuning the loss weights and learning rate in `scripts/config.py`.

## Inference and Transformation Testing

Run inference/evaluation:

```bash
python scripts/test.py
```

By default, `scripts/test.py` evaluates with `transformations = ['none']`.
To test robustness under benign edits, modify the `transformations` list in `scripts/test.py` (for example: `['none', 'noise', 'brightness', 'contrast', 'blur', 'flip', 'jpeg']`).

As with training, make sure `scripts/config.py` points to correct data/model paths and your desired hyperparameters.


## Citation

If you use this project, please cite our paper:

```bibtex
@article{
zhou2026metaseal,
title={MetaSeal: Defending Against Image Attribution Forgery Through Content-Dependent Cryptographic Watermarks},
author={Tong Zhou and Ruyi Ding and Gaowen Liu and Charles Fleming and Ramana Rao Kompella and Yunsi Fei and Xiaolin Xu and Shaolei Ren},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=8i3ErmCfdJ},
}

```

## Acknowledgements

This implementation benefited from ideas and/or codebases from prior watermarking and invertible-network work. We thank the authors and maintainers of:

- **HiNet** https://github.com/TomTomTommi/HiNet
- **FIN** https://github.com/QQiuyp/FIN
- **HiDDeN** https://github.com/ando-khachatryan/HiDDeN

