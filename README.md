<div align="center">

# HSR-Former
### Hybrid Sparse Range Attention Transformer for High-Quality Image Restoration

**Debashis Das · Shivam Kripashankar Singh · Suman Kumar Maji**  
Department of Computer Science & Engineering, Indian Institute of Technology Patna

[![Paper](https://img.shields.io/badge/IEEE%20TCE-Accepted-0A66C2?style=for-the-badge)](#citation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Tasks](https://img.shields.io/badge/Tasks-Denoising%20%7C%20LLIE%20%7C%20Deblurring-7B61FF?style=for-the-badge)](#supported-restoration-tasks)

**Official-style PyTorch repository for HSR-Former**, organized in the spirit of Restormer with separate task instructions, reproducible YAML configurations, training/testing scripts, a model-zoo interface, tiled inference, and Google Drive hooks for checkpoints and visual results.

</div>

---

## News

- **2026:** HSR-Former accepted for publication in *IEEE Transactions on Consumer Electronics*.
- **Code release:** training, evaluation, demo, task configurations, architecture assets, and reproducibility notes are included in this repository.
- **Model zoo / visual results:** Google Drive placeholders are already wired into the repository; replace them with your public links when weights and restored images are uploaded.

---

## Abstract

HSR-Former is a unified image-restoration Transformer designed to combine **fine local structural modeling** with **efficient long-range contextual aggregation**. It introduces: (1) a **Texture-Infused Feature Block (TIFB)** with fixed multi-directional gradient cues, (2) **Hybrid Sparse Local-Global Multi-Head Attention (HSLG-MHA)** combining **Spatial Feature Attention Unit (SFAU)** and **Sparse Region-Focused Multi-Head Attention (SRF-MHA)**, and (3) a **Multi-Branch Depthwise Feed-Forward Network (MBD-FFN)** for multi-scale token interaction. The same backbone is trained with independent task-specific weights for denoising, low-light enhancement, and motion deblurring.

---

## Network Architecture

<p align="center">
  <img src="assets/HSRFormer_architecture.png" width="100%" alt="HSR-Former Architecture">
</p>

### Sparse attention design

<p align="center">
  <img src="assets/HSRFormer_attention_comparison.png" width="100%" alt="Window, shifted-window and proposed sparse attention comparison">
</p>

### Core design at a glance

| Component | Role |
|---|---|
| **TIFB** | Combines shallow convolutional features with fixed four-direction gradient texture cues. |
| **TEB** | Uses horizontal, vertical, and two diagonal 3×3 gradient kernels. |
| **HSLG-MHA** | Splits channels into local and global paths and fuses SFAU + SRF-MHA responses. |
| **SFAU** | Builds a spatial saliency descriptor from channel-averaged local features and performs residual spatial gating. |
| **SRF-MHA** | Restricts attention to dilated sparse key/value neighborhoods, avoiding dense all-token attention. |
| **MBD-FFN** | Uses parallel depthwise kernels **1×1, 2×2, 3×3, 4×4, 5×5** followed by pointwise fusion and GELU. |
| **Hierarchy** | Four levels with feature widths **[48, 96, 192, 384]**, blocks **[4, 6, 6, 8]**, and heads **[1, 2, 4, 8]**. |
| **Sampling** | Pixel-unshuffle in the encoder and pixel-shuffle in the decoder with additive skip connections. |
| **Reconstruction** | Final 3×3 convolution + residual addition to the degraded input. |

The default reference configuration contains **14.740M trainable parameters**, aligned with the 14.74M parameter count reported in the manuscript. See [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) for manuscript-specified settings and explicitly documented low-level defaults.

---

## Supported Restoration Tasks

| Task | Training | Evaluation | Config | Pretrained Model | Visual Results |
|---|---|---|---|---|---|
| Gaussian grayscale denoising | [Instructions](Denoising/README.md#1-gaussian-grayscale-denoising) | [Instructions](Denoising/README.md#evaluation) | [`gaussian_gray.yml`](configs/denoising/gaussian_gray.yml) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) |
| Gaussian color denoising | [Instructions](Denoising/README.md#2-gaussian-color-denoising) | [Instructions](Denoising/README.md#evaluation) | [`gaussian_color.yml`](configs/denoising/gaussian_color.yml) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) |
| Real image denoising | [Instructions](Denoising/README.md#3-real-image-denoising) | [Instructions](Denoising/README.md#evaluation) | [`real_sidd.yml`](configs/denoising/real_sidd.yml) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) |
| Low-light enhancement (LOL-v1) | [Instructions](Low_Light_Enhancement/README.md#training) | [Instructions](Low_Light_Enhancement/README.md#evaluation) | [`lolv1.yml`](configs/low_light/lolv1.yml) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) |
| Low-light enhancement (FiveK) | [Instructions](Low_Light_Enhancement/README.md#mit-adobe-fivek) | [Instructions](Low_Light_Enhancement/README.md#mit-adobe-fivek) | [`fivek.yml`](configs/low_light/fivek.yml) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) |
| Motion deblurring (GoPro) | [Instructions](Motion_Deblurring/README.md#training) | [Instructions](Motion_Deblurring/README.md#evaluation) | [`gopro.yml`](configs/deblurring/gopro.yml) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) |
| Deblurring transfer (HIDE / RealBlur) | — | [Instructions](Motion_Deblurring/README.md#evaluation) | [`hide.yml`](configs/deblurring/hide.yml), [`realblur.yml`](configs/deblurring/realblur.yml) | GoPro-trained model | [Google Drive — add link](docs/GOOGLE_DRIVE_LINKS.md) |

---

## Installation

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/HSRFormer.git
cd HSRFormer

conda create -n hsrformer python=3.10 -y
conda activate hsrformer
pip install -r requirements.txt
```

For detailed setup and CUDA notes, see [`INSTALL.md`](INSTALL.md).

Quick verification:

```bash
python -m compileall hsrformer train.py test.py demo.py
pytest -q
```

---

## Dataset Layout

HSR-Former uses two generic data modes so the training code remains compact and reusable.

### Paired restoration datasets

Use the same filenames in degraded/input and clean/target folders:

```text
datasets/
└── low_light/LOLv1/
    ├── train/
    │   ├── low/
    │   │   ├── 0001.png
    │   │   └── ...
    │   └── high/
    │       ├── 0001.png
    │       └── ...
    └── test/
        ├── low/
        └── high/
```

The same convention is used for SIDD, GoPro, HIDE, RealBlur, and FiveK after adapting folder names in the YAML file.

### Synthetic Gaussian denoising

Store clean training images under:

```text
datasets/denoising/train/
├── BSD400/
├── DIV2K/
└── Flickr2K/
```

Gaussian noise is generated **on the fly**. The paper evaluates noise levels **σ = 30, 40, 50, 60**; these are already configured in the denoising YAML files.

---

## Training

The paper uses independent task-specific weights with the same HSR-Former architecture. The supplied configurations follow the reported training recipe:

- optimizer: **AdamW**
- β₁ = **0.9**, β₂ = **0.999**
- weight decay = **1×10⁻⁴**
- loss: **L1**
- total iterations: **320K**
- learning rate: **3×10⁻⁴ → 1×10⁻⁶** with cosine annealing
- crop size: **128×128**
- augmentation: random rotation + horizontal flip

### Single GPU

```bash
python train.py --config configs/denoising/gaussian_color.yml
```

or

```bash
./train.sh configs/denoising/gaussian_color.yml 1
```

### Multi-GPU with `torchrun`

```bash
./train.sh configs/denoising/gaussian_color.yml 4
```

Equivalent:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/denoising/gaussian_color.yml
```

### Resume training

```bash
python train.py \
  --config configs/denoising/gaussian_color.yml \
  --resume experiments/gaussian_color/latest.pth
```

Checkpoints are written to the `experiment_dir` defined by each YAML file.

---

## Evaluation

### Gaussian color denoising

```bash
python test.py \
  --config configs/denoising/gaussian_color.yml \
  --checkpoint pretrained_models/hsrformer_gaussian_color.pth \
  --output-dir results/gaussian_color \
  --save-images
```

### Real image denoising

```bash
python test.py \
  --config configs/denoising/real_sidd.yml \
  --checkpoint pretrained_models/hsrformer_real_sidd.pth \
  --output-dir results/sidd \
  --save-images
```

### Low-light enhancement

```bash
python test.py \
  --config configs/low_light/lolv1.yml \
  --checkpoint pretrained_models/hsrformer_lolv1.pth \
  --output-dir results/lolv1 \
  --save-images
```

### Motion deblurring

```bash
python test.py \
  --config configs/deblurring/gopro.yml \
  --checkpoint pretrained_models/hsrformer_gopro.pth \
  --output-dir results/gopro \
  --save-images
```

### High-resolution / memory-limited evaluation

Use overlapping tiled inference:

```bash
python test.py \
  --config configs/deblurring/realblur.yml \
  --checkpoint pretrained_models/hsrformer_gopro.pth \
  --tile 512 \
  --tile-overlap 64 \
  --save-images
```

The test script reports per-image and average **PSNR / SSIM** and optionally saves restored PNG files.

For **LPIPS** (paired low-light) and **NIQE** (no-reference low-light) evaluation, install the optional perceptual dependencies and run [`scripts/evaluate_perceptual.py`](scripts/evaluate_perceptual.py).

---

## Demo on Your Own Images

```bash
python demo.py \
  --task Low_Light_Enhancement \
  --checkpoint pretrained_models/hsrformer_lolv1.pth \
  --input demo/degraded \
  --result-dir demo/restored
```

Available task names:

```text
Gaussian_Gray_Denoising
Gaussian_Color_Denoising
Real_Denoising
Low_Light_Enhancement
Motion_Deblurring
```

---

## Paper Results

### Real denoising

| Dataset | PSNR | SSIM |
|---|---:|---:|
| SIDD | **40.19** | **0.966** |
| PolyU | **40.25** | **0.973** |
| Nam | **40.40** | **0.990** |

### Low-light enhancement

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| LOL-v1 | **24.48** | **0.868** | **0.119** |
| MIT-Adobe FiveK | **24.79** | **0.927** | **0.060** |

### Motion deblurring

| Dataset | PSNR | SSIM |
|---|---:|---:|
| GoPro | **33.25** | **0.965** |
| HIDE | **31.37** | **0.949** |
| RealBlur | **32.70** | **0.932** |

### Complexity reported in the paper

| Params | MACs | PolyU inference time | PolyU PSNR / SSIM |
|---:|---:|---:|---:|
| **14.74M** | **47.41G** | **0.18 s** | **40.25 / 0.973** |

> Reproducing exact benchmark numbers requires the authors' released checkpoints, identical dataset preprocessing, and the evaluation protocol used for the manuscript.

The paper also includes a YOLOv8 downstream experiment on ExDark. Because the manuscript specifies the 80:20 split but does not fully enumerate the detector training hyperparameters, this repository intentionally does not invent an exact detector recipe; restored ExDark images can be generated with `demo.py` and evaluated using the detector configuration used by the authors.

---

## Google Drive: Checkpoints and Visual Outputs

This repository includes a ready-to-fill model/visual-results manifest:

```text
docs/GOOGLE_DRIVE_LINKS.md
model_zoo.json
```

After uploading files or folders to Google Drive, replace the placeholders in `model_zoo.json`. Users can then download a file or entire folder with:

```bash
python scripts/download_gdrive.py --url "<GOOGLE_DRIVE_URL>" --output pretrained_models/model.pth
```

For a shared folder:

```bash
python scripts/download_gdrive.py --url "<GOOGLE_DRIVE_FOLDER_URL>" --output visual_results --folder
```

Recommended Drive structure:

```text
HSRFormer_Release/
├── pretrained_models/
│   ├── hsrformer_gaussian_gray.pth
│   ├── hsrformer_gaussian_color.pth
│   ├── hsrformer_real_sidd.pth
│   ├── hsrformer_lolv1.pth
│   ├── hsrformer_fivek.pth
│   └── hsrformer_gopro.pth
└── visual_results/
    ├── gaussian_gray/
    ├── gaussian_color/
    ├── real_denoising/
    ├── low_light/
    └── deblurring/
```

---

## Repository Structure

```text
HSRFormer/
├── README.md
├── INSTALL.md
├── train.py
├── test.py
├── demo.py
├── train.sh
├── model_zoo.json
├── requirements.txt
├── assets/
│   ├── HSRFormer_architecture.png
│   └── HSRFormer_attention_comparison.png
├── hsrformer/
│   ├── models/hsrformer.py
│   ├── data/datasets.py
│   └── utils/
├── configs/
│   ├── denoising/
│   ├── low_light/
│   └── deblurring/
├── Denoising/README.md
├── Low_Light_Enhancement/README.md
├── Motion_Deblurring/README.md
├── scripts/
├── pretrained_models/
├── docs/
└── tests/
```

---

## Reproducibility Notes

The manuscript specifies the high-level architecture and training protocol, but does not numerically define every implementation-level choice needed to execute code. The repository therefore **does not hide those choices**. They are centralized and documented in [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md), including:

- local/global channel split ratio;
- exact sparse neighborhood kernel size;
- mapping of dilation rates to heads when the number of heads exceeds three;
- MBD-FFN internal projection ratio;
- treatment of the deepest encoder/decoder notation in the architecture diagram;
- refinement depth.

The defaults are selected to be consistent with the figure, equations, reported stage configuration, and the reported **14.74M** parameter count.

---

## Citation

If this repository or HSR-Former is useful in your research, please cite:

```bibtex
@article{das2026hsrformer,
  title   = {HSR-Former: Hybrid Sparse Range Attention Transformer for High-Quality Image Restoration},
  author  = {Das, Debashis and Singh, Shivam Kripashankar and Maji, Suman Kumar},
  journal = {IEEE Transactions on Consumer Electronics},
  year    = {2026},
  note    = {Accepted}
}
```

Update the BibTeX with volume, issue, pages, and DOI when IEEE publishes the final bibliographic record.

---

## Contact

For questions related to the method or official checkpoints, please open a GitHub issue or contact the paper authors through their institutional contact details.

---

## Acknowledgment

The repository layout is inspired by the clean task-oriented release style of **Restormer**. The HSR-Former model implementation, task configurations, and training/evaluation code in this repository are organized specifically around the architecture and experiments described in the HSR-Former manuscript.
