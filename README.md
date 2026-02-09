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


## Citation

If you use this project, please cite our paper (arxiv version, TMLR version will be updated later):

```bibtex
@article{zhou2025content,
  title={A Content-dependent Watermark for Safeguarding Image Attribution},
  author={Zhou, Tong and Ding, Ruyi and Liu, Gaowen and Fleming, Charles and Kompella, Ramana Rao and Fei, Yunsi and Xu, Xiaolin and Ren, Shaolei},
  journal={arXiv preprint arXiv:2509.10766},
  year={2025}
}

```

## Acknowledgements

This implementation benefited from ideas and/or codebases from prior watermarking and invertible-network work. We thank the authors and maintainers of:

- **HiNet** https://github.com/TomTomTommi/HiNet
- **FIN** https://github.com/QQiuyp/FIN
- **HiDDeN** https://github.com/ando-khachatryan/HiDDeN


