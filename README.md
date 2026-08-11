<div align="center">

# HSR-Former
## Hybrid Sparse Range Attention Transformer for High-Quality Image Restoration

**Accepted paper — IEEE Transactions on Consumer Electronics**

**Debashis Das · Shivam Kripashankar Singh · Suman Kumar Maji**  
Department of Computer Science and Engineering, Indian Institute of Technology Patna, India

[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tasks](https://img.shields.io/badge/Tasks-Denoising%20%7C%20Low--Light%20%7C%20Deblurring-6C63FF)](#supported-restoration-tasks)
[![Parameters](https://img.shields.io/badge/Parameters-14.74M-success)](#architecture)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**[Paper: add final IEEE link] · [Pretrained Models](model_zoo.json) · [Visual Results](visual_results.json) · [Installation](INSTALL.md) · [Reproducibility](docs/REPRODUCIBILITY.md)**

</div>

---

## Overview

HSR-Former is a hierarchical image-restoration Transformer designed to combine **fine local structural modeling** with **sparse long-range spatial interaction**. The network contains:

- **TIFB** — Texture-Infused Feature Block.
- **TEB** — fixed multi-directional gradient texture extraction.
- **HSLG-MHA** — Hybrid Sparse Local-Global Multi-Head Attention.
- **SFAU** — local spatial feature attention branch.
- **SRF-MHA** — head-wise dilated sparse region-focused attention branch.
- **MBD-FFN** — five-branch depthwise feed-forward network with kernel sizes `1×1` through `5×5`.
- Four-level pixel-unshuffle / pixel-shuffle encoder-decoder with additive skips and residual reconstruction.

The paper evaluates one architecture across **synthetic denoising, real denoising, low-light enhancement, and motion deblurring**, while training task-specific weights.

<p align="center">
  <img src="assets/HSRFormer_architecture.png" width="100%" alt="HSR-Former architecture">
</p>

<p align="center"><b>HSR-Former architecture.</b></p>

<p align="center">
  <img src="assets/HSRFormer_attention_comparison.png" width="95%" alt="Attention comparison">
</p>

---

## News

- **2026** — HSR-Former accepted for publication in *IEEE Transactions on Consumer Electronics*.
- **Code release** — This repository provides independent data-preparation, path-generation, training, testing, metric, demo, and release utilities for all tasks reported in the manuscript.

---

## Supported restoration tasks

<table>
<tr>
  <th>Task</th><th>Training</th><th>Testing</th><th>Data preparation</th><th>Visual results</th>
</tr>
<tr>
  <td>Gaussian grayscale denoising</td>
  <td><a href="Denoising/README.md#5-train-synthetic-grayscale-denoising">Train</a></td>
  <td><a href="Denoising/README.md#8-test-gaussian-grayscale-denoising">Test</a></td>
  <td><a href="Denoising/README.md#2-prepare-clean-synthetic-denoising-data">Prepare</a></td>
  <td>Google Drive placeholder in <code>visual_results.json</code></td>
</tr>
<tr>
  <td>Gaussian color denoising</td>
  <td><a href="Denoising/README.md#6-train-synthetic-color-denoising">Train</a></td>
  <td><a href="Denoising/README.md#9-test-gaussian-color-denoising">Test</a></td>
  <td><a href="Denoising/README.md#2-prepare-clean-synthetic-denoising-data">Prepare</a></td>
  <td>Google Drive placeholder in <code>visual_results.json</code></td>
</tr>
<tr>
  <td>Real image denoising</td>
  <td><a href="Denoising/README.md#7-train-real-denoising-on-sidd">Train</a></td>
  <td><a href="Denoising/README.md#10-test-real-denoising">Test</a></td>
  <td><a href="Denoising/README.md#3-prepare-real-noise-data">Prepare</a></td>
  <td>Google Drive placeholder in <code>visual_results.json</code></td>
</tr>
<tr>
  <td>Low-light enhancement</td>
  <td><a href="Low_Light_Enhancement/README.md#6-train-lol-v1">Train</a></td>
  <td><a href="Low_Light_Enhancement/README.md#8-test-lol-v1">Test</a></td>
  <td><a href="Low_Light_Enhancement/README.md#2-prepare-lol-v1">Prepare</a></td>
  <td>Google Drive placeholder in <code>visual_results.json</code></td>
</tr>
<tr>
  <td>Motion deblurring</td>
  <td><a href="Motion_Deblurring/README.md#6-train-on-gopro">Train</a></td>
  <td><a href="Motion_Deblurring/README.md#7-test-gopro">Test</a></td>
  <td><a href="Motion_Deblurring/README.md#2-prepare-gopro">Prepare</a></td>
  <td>Google Drive placeholder in <code>visual_results.json</code></td>
</tr>
</table>

---

## Architecture

The default model uses the manuscript-level settings:

```text
Feature dimensions : [48, 96, 192, 384]
Transformer blocks : [ 4,  6,   6,   8]
Attention heads    : [ 1,  2,   4,   8]
Encoder            : Pixel-Unshuffle
Decoder            : Pixel-Shuffle
Loss               : L1
Optimizer          : AdamW
Training iterations: 320,000
Patch size         : 128 × 128
Initial LR         : 3e-4
Final LR           : 1e-6
Scheduler          : Cosine annealing
```

Default executable parameter count:

```bash
python scripts/model_info.py
```

```text
14,740,159 parameters ≈ 14.74M
```

The paper reports **14.74M parameters and 47.41 GMACs** for the proposed model. See [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md) for the equation/module mapping and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for implementation-level parameters not numerically fixed by the manuscript.

---

## Repository structure

```text
HSRFormer/
├── README.md
├── INSTALL.md
├── train.py                         # generic trainer
├── test.py                          # generic single-config evaluator
├── demo.py                          # restore your own images
├── model_zoo.json                   # Google Drive weight links
├── visual_results.json              # Google Drive visual-result links
│
├── hsrformer/
│   ├── models/
│   │   └── hsrformer.py             # complete architecture
│   ├── data/
│   │   ├── datasets.py              # paired/Gaussian/unpaired datasets
│   │   └── path_utils.py            # pairing + manifest utilities
│   ├── engine/
│   │   └── evaluation.py
│   └── utils/
│
├── Denoising/
│   ├── prepare_data.py              # synthetic + SIDD + paired datasets
│   ├── generate_paths.py            # manifest generation
│   ├── train_gaussian_gray.py
│   ├── train_gaussian_color.py
│   ├── train_real.py
│   ├── test_gaussian_gray.py
│   ├── test_gaussian_color.py
│   ├── test_real.py
│   ├── Options/
│   └── README.md
│
├── Low_Light_Enhancement/
│   ├── prepare_data.py              # LOL-v1/FiveK/no-reference
│   ├── generate_paths.py
│   ├── train_lolv1.py
│   ├── train_fivek.py
│   ├── test_lolv1.py
│   ├── test_fivek.py
│   ├── test_noref.py                # DICM/LIME/MEF/NPE + NIQE
│   ├── Options/
│   └── README.md
│
├── Motion_Deblurring/
│   ├── prepare_data.py              # GoPro/HIDE/RealBlur
│   ├── generate_paths.py
│   ├── train_gopro.py
│   ├── test_gopro.py
│   ├── test_hide.py
│   ├── test_realblur_j.py
│   ├── test_realblur_r.py
│   ├── Options/
│   └── README.md
│
├── datasets/
│   └── manifests/
├── pretrained_models/
├── results/
├── scripts/
├── docs/
└── tests/
```

---

## Installation

```bash
git clone https://github.com/<YOUR-USERNAME>/HSRFormer.git
cd HSRFormer

conda create -n hsrformer python=3.10 -y
conda activate hsrformer

pip install -r requirements.txt
pip install -e .
```

For LPIPS and NIQE:

```bash
pip install -r requirements-perceptual.txt
```

Full details: [`INSTALL.md`](INSTALL.md).

---

# End-to-end reproduction

## A. Denoising

### A1. Prepare datasets

```bash
python Denoising/prepare_data.py clean --source /data/BSD400   --name BSD400   --split train
python Denoising/prepare_data.py clean --source /data/DIV2K    --name DIV2K    --split train
python Denoising/prepare_data.py clean --source /data/Flickr2K --name Flickr2K --split train
python Denoising/prepare_data.py sidd --source /data/SIDD_Medium_Srgb --split train
```

Add test datasets using the same script, then generate manifests:

```bash
python Denoising/generate_paths.py
```

### A2. Train

```bash
python Denoising/train_gaussian_gray.py
python Denoising/train_gaussian_color.py
python Denoising/train_real.py
```

### A3. Test

```bash
python Denoising/test_gaussian_gray.py  --checkpoint pretrained_models/hsrformer_gaussian_gray.pth --dataset all --sigma all
python Denoising/test_gaussian_color.py --checkpoint pretrained_models/hsrformer_gaussian_color.pth --dataset all --sigma all
python Denoising/test_real.py           --checkpoint pretrained_models/hsrformer_real_sidd.pth --dataset all
```

Detailed instructions: [`Denoising/README.md`](Denoising/README.md).

---

## B. Low-light enhancement

### B1. Prepare LOL-v1

```bash
python Low_Light_Enhancement/prepare_data.py lolv1 --source /data/LOL-v1
python Low_Light_Enhancement/generate_paths.py
```

### B2. Train

```bash
python Low_Light_Enhancement/train_lolv1.py
```

### B3. Test paired metrics

```bash
python Low_Light_Enhancement/test_lolv1.py \
  --checkpoint pretrained_models/hsrformer_lolv1.pth \
  --lpips
```

### B4. Test no-reference NIQE

```bash
python Low_Light_Enhancement/test_noref.py \
  --checkpoint pretrained_models/hsrformer_lolv1.pth \
  --dataset DICM \
  --niqe
```

Detailed instructions: [`Low_Light_Enhancement/README.md`](Low_Light_Enhancement/README.md).

---

## C. Motion deblurring

### C1. Prepare GoPro

```bash
python Motion_Deblurring/prepare_data.py gopro --source /data/GoPro
python Motion_Deblurring/generate_paths.py
```

### C2. Train

```bash
python Motion_Deblurring/train_gopro.py
```

### C3. Test

```bash
python Motion_Deblurring/test_gopro.py      --checkpoint pretrained_models/hsrformer_gopro.pth
python Motion_Deblurring/test_hide.py       --checkpoint pretrained_models/hsrformer_gopro.pth
python Motion_Deblurring/test_realblur_j.py --checkpoint pretrained_models/hsrformer_gopro.pth
python Motion_Deblurring/test_realblur_r.py --checkpoint pretrained_models/hsrformer_gopro.pth
```

Detailed instructions: [`Motion_Deblurring/README.md`](Motion_Deblurring/README.md).

---

## Multi-GPU training

All tasks use the same distributed trainer:

```bash
torchrun --nproc_per_node=4 train.py \
  --config Denoising/Options/GaussianColor_HSRFormer.yml
```

Change the YAML path for low-light/deblurring.

---

## Tiled testing for large images

Testing automatically pads non-multiple-of-8 inputs. Large images can additionally be tiled:

```bash
python Motion_Deblurring/test_realblur_j.py \
  --checkpoint pretrained_models/hsrformer_gopro.pth \
  --tile 512 \
  --tile-overlap 64
```

The same options are available for real denoising and low-light testing.

---

## Demo on your own image/folder

```bash
python demo.py \
  --task Motion_Deblurring \
  --checkpoint pretrained_models/hsrformer_gopro.pth \
  --input demo/degraded \
  --result-dir demo/restored
```

Supported task names:

```text
Gaussian_Gray_Denoising
Gaussian_Color_Denoising
Real_Denoising
Low_Light_Enhancement
Motion_Deblurring
```

---

## Paper results

The accepted manuscript reports the following representative results.

### Real denoising

| Dataset | PSNR | SSIM |
|---|---:|---:|
| SIDD | **40.19** | **0.966** |
| PolyU | **40.25** | **0.973** |
| Nam | **40.40** | **0.990** |

### Low-light enhancement

| Dataset | PSNR | SSIM | LPIPS ↓ |
|---|---:|---:|---:|
| LOL-v1 | **24.48** | **0.868** | **0.119** |
| MIT-Adobe FiveK | **24.79** | **0.927** | **0.060** |

No-reference NIQE reported by the paper: DICM **3.71**, LIME **3.69**, MEF **3.74**, NPE **3.80**.

### Motion deblurring

| Dataset | PSNR | SSIM |
|---|---:|---:|
| GoPro | **33.25** | **0.965** |
| HIDE | **31.37** | **0.949** |
| RealBlur | **32.70** | **0.932** |

---

## Pretrained weights and visual results via Google Drive

Large artifacts should not be committed to GitHub. Add your released URLs in:

```text
model_zoo.json
visual_results.json
```

Recommended Drive structure and download instructions are provided in [`docs/GOOGLE_DRIVE.md`](docs/GOOGLE_DRIVE.md).

---

## Configuration files

Every task has its own YAML file. For example:

```text
Denoising/Options/GaussianGray_HSRFormer.yml
Denoising/Options/GaussianColor_HSRFormer.yml
Denoising/Options/RealDenoising_HSRFormer.yml
Low_Light_Enhancement/Options/LOLv1_HSRFormer.yml
Low_Light_Enhancement/Options/FiveK_HSRFormer.yml
Motion_Deblurring/Options/GoPro_HSRFormer.yml
```

This makes batch size, data manifests, dilation schedule, model width, training iterations and optimizer settings explicit and editable.

---

## Citation

Please update DOI/volume/pages when the final IEEE bibliographic record becomes available.

```bibtex
@article{das2026hsrformer,
  title   = {HSR-Former: Hybrid Sparse Range Attention Transformer for High-Quality Image Restoration},
  author  = {Das, Debashis and Singh, Shivam Kripashankar and Maji, Suman Kumar},
  journal = {IEEE Transactions on Consumer Electronics},
  year    = {2026},
  note    = {Accepted for publication}
}
```

---

## Reproducibility note

The code directly follows the architecture shown in the manuscript and exposes manuscript-unspecified low-level parameters rather than silently presenting them as published constants. Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before releasing checkpoints intended for exact paper reproduction.

---

## Contact

For questions about the paper or released checkpoints, open a GitHub issue or add the corresponding author email after the repository is published.

---

## Acknowledgment

The repository organization is inspired by well-maintained image-restoration codebases such as Restormer, with task-specific preparation/training/testing instructions and external hosting for large pretrained models and visual results.
