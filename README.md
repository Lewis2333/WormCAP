# High-Throughput Behavioral Phenotyping Framework for *C. elegans*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.5+-orange.svg)](https://www.paddlepaddle.org.cn/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**From Segmentation to Tracking and Multi-Dimensional Quantification**

[Paper](#citation) | [Installation](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation)

</div>

---

## Overview

This repository contains the official implementation of **"High-Throughput Behavioral Phenotyping Cascade Framework for *Caenorhabditis elegans*"**.

The system provides an automated, end-to-end solution for nematode behavioral analysis, effectively addressing challenges including non-rigid deformation, dense aggregation, and identity switches through a **"Segmentation → Tracking → Quantification"** cascade pipeline.

---

## Key Features

### 1. Robust Segmentation (Mask-RT-DETR)
- Transformer-based architecture for handling complex backgrounds and overlapping worms
- Outperforms Mask R-CNN in both precision (**+6% AP**) and inference latency (**-20%**)
- End-to-end prediction without NMS post-processing

### 2. Enhanced Tracking (PCA + Temporal Context)
- **PCA Morphological Constraints**: Eigenvalue-based shape confidence score (S_shape) to suppress non-worm noise
- **Temporal Context Recovery**: Kalman filter predictions with Mahalanobis distance matching for low-contrast scenarios
- **BoT-SORT Integration**: Robust multi-object tracking with camera motion compensation (CMC)

### 3. Bio-Kinematic Quantification
- **Pharynx-Anchored Logic**: Defines body axis using pharynx (10% skeletal position) to reduce head-sway noise
- **Deformation Analysis**:
  - **Stretch Ratio (Rs)**: Body extension/contraction dynamics
  - **Asymmetry (Ad)**: Bilateral imbalance detection via ray-marching
  - **Segmental Curvature (Ks)**: 10-segment fine-grained bending analysis
  - **Deformation Gradient (Gd)**: Curvature variation intensity
- **Event Detection**: Automatic classification of Omega Turns, Forward, Backward, and Stationary states

---

## Project Structure

```
WormTracker_Fusion/
│
├── main.py                  # CLI entry point for video processing
├── requirements.txt         # Dependency list
├── README.md                # Documentation
│
├── core/                    # Core Algorithms
│   ├── tracker.py           # VideoTracker (Mask-RT-DETR + BoT-SORT)
│   ├── features.py          # Skeleton extraction, PCA, Curvature, Asymmetry
│   └── kalman.py            # Kalman Filters for Pharynx/Tail/Center
│
└── utils/                   # Utilities
    ├── filters.py           # Signal processing (IQR, Z-score, Savitzky-Golay)
    ├── visualization.py     # OpenCV drawing tools
    └── io_utils.py          # Data export (CSV/JSON)
```

---

## <a id="installation"></a>Installation

### 1. Clone the repository

```bash
git clone https://github.com/your_username/WormTracker_Fusion.git
cd WormTracker_Fusion
```

### 2. Install dependencies

It is recommended to use a Conda environment (Python 3.8+).

```bash
# Create environment
conda create -n worm_pheno python=3.9
conda activate worm_pheno

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```txt
paddlepaddle-gpu>=2.5.0    # or paddlepaddle (CPU version)
paddlex>=3.0.0
ultralytics>=8.0.196
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.21.0
networkx>=3.1
filterpy>=1.4.5
pandas>=2.0.0
pyyaml>=6.0
```

*Note: Ensure you have the correct version of `paddlex` and `ultralytics` compatible with your CUDA version.*

---

## <a id="quick-start"></a>Usage

Run the `main.py` script to process a video. You must provide the path to your trained PaddleX inference model.

### Basic Command

```bash
python main.py --video ./data/sample.avi --model_dir ./models/mask_rt_detr_infer
```

### Full Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | **Required** | Path to the input video file |
| `--model_dir` | str | **Required** | Path to the exported PaddleX inference model folder |
| `--output_dir` | str | `output` | Directory to save CSVs, JSONs, and video results |
| `--conf` | float | `0.01` | Detection confidence threshold |
| `--save_video` | flag | `False` | If set, saves the visualized tracking video (.mp4) |
| `--show` | flag | `False` | If set, displays a real-time window (slower, for debug) |

### Example: Processing with Video Save

```bash
python main.py \
  --video ./datasets/N2_wildtype.avi \
  --model_dir ./weights/inference_model \
  --output_dir ./results/experiment_01 \
  --save_video \
  --conf 0.1
```

---

## Output Data

The system generates the following files in the `output_dir`:

### 1. `per_frame_features.csv`

A high-resolution log of every tracked worm in every frame. Columns include:

| Column | Description |
|--------|-------------|
| `track_id` | Unique identifier |
| `frame_id` | Frame number |
| `time_s` | Timestamp (seconds) |
| `speed`, `speed_forward`, `speed_backward` | Velocity metrics (pixels/s) |
| `length`, `width`, `area` | Geometric features |
| `overall_curvature` | Global bending metric (C) |
| `curvature_seg_1` ~ `curvature_seg_10` | Local curvature for 10 body segments (Ks) |
| `stretch_ratio` | Body extension ratio (Rs) |
| `asymmetry` | Deformation asymmetry score (Ad) |
| `deformation_gradient` | Curvature variation (Gd) |
| `omega_turn` | Binary flag (1 = Omega Turn, 0 = Normal) |
| `reversal` | Binary flag for reversal events |
| `pharynx_position`, `tail_position` | Coordinate pairs (x, y) |

### 2. `summary.json`

Aggregated statistics for each tracked individual:

```json
{
  "1": {
    "length_mean": 50.24,
    "speed_mean": 12.34,
    "stretch_ratio_mean": 1.05,
    "asymmetry_mean": 1.25,
    "omega_turn_count": 1,
    "reversal_count": 8
  }
}
```

### 3. `tracked_video.mp4` (Optional)

Visualized output with:
- Bounding boxes & track IDs
- Skeleton centerlines (green)
- Pharynx markers (red, 10% position)
- Tail markers (blue)
- Trajectory trails (optional)

---

## <a id="documentation"></a>Methodology Overview

### Pipeline

The framework follows a **"Segmentation–Tracking–Quantification"** cascade:

1. **Segmentation**
   - `Mask-RT-DETR` predicts pixel-level masks and bounding boxes
   - Transformer encoder-decoder with Hungarian matching loss

2. **Refinement**
   - **PCA Constraint**: Validates mask shape using major/minor axis eigenvalues (λ₁, λ₂)
   - **Skeletonization**: Zhang-Suen algorithm + graph-based branch pruning (NetworkX)

3. **Tracking**
   - Hungarian matching based on **IoU + Motion + Shape Consistency**
   - **Pharynx Kinematics**: Speed calculated via projection along pharynx-tail axis
   - Two-stage association (high/low confidence cascade)

4. **Smoothing**
   - B-Spline interpolation for centerline
   - IQR + Savitzky-Golay filtering to remove jitter

### Feature Stability (Wild-Type N2, n=5)

| Feature | Mean ± SD | CV (%) | Biological Validity |
|---------|-----------|--------|---------------------|
| Body Length | 50.24 ± 2.31 px | 7.0 | ✅ Geometric conservation |
| Extension Ratio (Rs) | 1.00 ± 0.05 | 4.5 | ✅ Stable regulation (0.91-1.09) |
| Asymmetry (Ad) | 1.25 ± 0.19 px | 15.2 | ✅ Below 2.0px threshold |
| Global Curvature (C) | 1.08 ± 0.07 | 6.5 | ✅ Consistent sinusoidal posture |

---

## Datasets

### Training Data
- **CSB-1**: 10 videos (912×736px, 5 FPS), 8:1:1 train/val/test split
- **WormSwin Synthetic**: 10,000 images (175K+ instances) for pre-training

### Validation Data
- **Tierpsy Tracker (N2)**: 2 FPS, 370s recordings for long-term tracking evaluation

*All datasets follow COCO annotation format.*

---

## <a id="citation"></a>Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{Liu2025HighThroughput,
  title={High-Throughput Behavioral Phenotyping Cascade Framework for C. elegans: 
         From Segmentation to Tracking and Multi-Dimensional Quantification},
  author={Liu, Xiaoke and Li, Boao and Huo, Jing and Han, Xiaoqing},
  journal={Submitted to Journal Name},
  year={2025},
  affiliation={Shandong Second Medical University}
}
```

---

## Acknowledgments

This work was supported by:
- **Natural Science Foundation of Shandong Province** (Grant No. ZR2024QF228 and ZR2024QA176)
- **National Natural Science Foundation of China** (Grant No. 82301666)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

- **Corresponding Author**: Xiaoqing Han (hanxiaoqing@sdsmu.edu.cn)
- **Institution**: Shandong Second Medical University
- **Issues**: [GitHub Issues](https://github.com/Lewis2333/WormCap/issues)

---

<div align="center">
  
</div>
