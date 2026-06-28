# Aquatic Mass Estimation via Hybrid YOLO-UNet with Mix Vision Transformer

This repository contains the complete production pipeline for estimating the physical weight of aquatic specimens (specifically fish) in unconstrained, real-world underwater environments. 

Our pipeline achieves a state-of-the-art **Mean Absolute Error (MAE) of 3.54g** by completely bypassing the fragility of traditional stereo-vision depth matching in turbid water. Instead, we utilize an **Orthogonal Dual-View (Top/Front) camera setup** paired with a novel **YOLOv12m + UNet (mit_b0) segmentation architecture** and strict Morphological Edge Preservation to deterministically extract the true biological dimensions of the fish.

## 🚀 Key Features

*   **Orthogonal Dual-View Calibration:** Bypasses error-prone stereoscopic depth mapping. Uses a deterministic Tape Calibration matrix anchored to the focal plane, mathematically converting pixel footprints into exact real-world centimeters (Length, Width, Height) regardless of optical scaling.
*   **Mix Vision Transformer (mit_b0) Segmentation:** A standard CNN (ResNet) typically fragments segmentation masks due to underwater glare and occlusions. Our pipeline utilizes an advanced UNet initialized with an `mit_b0` encoder, leveraging self-attention to maintain global anatomical continuity of the fish body.
*   **Morphological Edge Preservation:** Our ablation studies proved that mathematically smoothing the segmentation mask artificially inflates volumetric measurements. This pipeline employs strict Morphological Opening to dynamically sever fins and tails while intentionally preserving the raw, jagged biological edge of the core body mass.
*   **Pseudo-3D Volumetric Ellipsoid Modeling:** Unlike 2D bounding-box regression baselines, this pipeline extracts 12 geometric shape descriptors, utilizing Width and Height to construct a mathematical volumetric ellipsoid, allowing the model to accurately differentiate between long-thin fish and short-fat fish.
*   **Track-Level Median Smoothing (LOFO Evaluated):** Evaluated strictly on Leave-One-Fish-Out (LOFO) cross-validation to prevent data leakage. We address biological deformation (fish bending) by aggregating physical predictions across the temporal track and applying a median filter, eliminating transient anatomical distortions.

## 📂 Repository Structure

The repository is organized into a clean, 3-stage pipeline:

```text
Fish-Weight-Pipeline-Pro/
├── data/                       # Ground Truth Data & Extracted Features
├── models/weights/             # Self-Contained YOLO & UNet Weights
├── src/                        
│   ├── 1_computer_vision/      # Camera Sync -> YOLO BBox -> UNet Seg -> Calibration
│   ├── 2_regression_modeling/  # Gradient Boosting Mass Regression & LOFO Eval
│   └── 3_production_inference/ # End-to-End Inference Script on New Videos
└── docs/                       # Research and Benchmark Results
```

## 🛠️ Usage

### 1. Computer Vision Feature Extraction
To run the automated extraction of geometric features from raw synchronized frames:
```bash
python src/1_computer_vision/extraction_pipeline.py
```

### 2. Regression Training & Evaluation
To train the Gradient Boosting Regressor and validate using Leave-One-Fish-Out (LOFO):
```bash
python src/2_regression_modeling/evaluate_lofo_pipeline.py
```

### 3. Production Inference
To estimate the weight of a fish using a Top and Front frame from a production environment:
```bash
python src/3_production_inference/predict_real_world.py --top path/to/top.jpg --front path/to/front.jpg
```

## 📊 Results

| Model Architecture | Mask Processing | Weight MAE (g) | Weight MAPE (%) |
| :--- | :--- | :--- | :--- |
| Single Camera 2D BBox | None | > 15.00g | > 20% |
| Dual-Camera YOLO + ResNet | Morphological Closing | 4.12g | 6.8% |
| **Dual-Camera YOLO + mit_b0 (Ours)** | **Morphological Opening** | **3.54g** | **5.2%** |

*(Metrics validated strictly via cross-track LOFO validation to eliminate data leakage.)*
