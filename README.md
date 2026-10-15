# 🦷 Dental Caries Detection and Lesion Segmentation using UNet++

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohitinnani004-afk/dental-caries-detection-unetpp/blob/main/notebooks/dental_caries_colab_training.ipynb)
[![Architecture](https://img.shields.io/badge/Architecture-UNet%2B%2B%20(Nested%20U--Net)-brightgreen.svg)](https://arxiv.org/abs/1807.10165)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, clinical-grade deep learning solution for automated detection, segmentation, and anatomical severity classification of **dental caries (tooth decay / cavities)** from digital dental radiographs (bitewing and periapical X-rays) using **UNet++ (Nested U-Net)** with **Deep Supervision** and **Spatial Attention Gates**.

---

## 📌 Table of Contents
1. [Overview & Clinical Motivation](#-overview--clinical-motivation)
2. [Why UNet++ for Dental Radiographs?](#-why-unet-for-dental-radiographs)
3. [Architecture & Mathematical Formulation](#-architecture--mathematical-formulation)
4. [Radiographic Caries Classification System](#-radiographic-caries-classification-system)
5. [Interactive Clinical Web Dashboard](#-interactive-clinical-web-dashboard)
6. [Project Structure](#-project-structure)
7. [Quick Start & Installation](#-quick-start--installation)
8. [Training Pipeline](#-training-pipeline)
9. [Evaluation & Clinical Reporting](#-evaluation--clinical-reporting)
10. [Inference & Lesion Extraction](#-inference--lesion-extraction)
11. [Unit & Integration Testing](#-unit--integration-testing)
12. [References](#-references)

---

## 🔬 Overview & Clinical Motivation

Dental caries is one of the most prevalent chronic conditions worldwide. Early detection on dental radiographs (bitewing, panoramic, and periapical scans) is critical to prevent irreversible pulpitis, root canal therapy, or tooth extraction.

However, radiographic caries detection presents severe computer vision challenges:
- **Low Contrast:** Demineralization creates faint radiolucent (dark) regions with soft, gradual boundaries against mineralized enamel and dentin.
- **Small Lesion Size & Class Imbalance:** Caries lesions often occupy less than **1% to 3%** of total radiograph pixels.
- **Confounding Anatomical Structures:** Overlapping enamel contact points, cervical burnout, dental restorations (amalgam/composite), and jawbone trabeculae frequently lead to false positives.

This project implements **UNet++ (Nested U-Net)** to address these challenges through dense multi-scale skip pathways, deep supervision, and automated lesion grading.

---

## 🧠 Why UNet++ for Dental Radiographs?

In standard U-Net, feature maps from the encoder are directly concatenated with the decoder. This introduces a **semantic gap** because early encoder layers contain low-level, high-resolution edge details, whereas decoder layers represent high-level semantic concepts.

### UNet++ Redesigns the Skip Pathway:
1. **Nested Dense Convolutional Nodes ($X^{i,j}$):** Intermediate dense blocks bridge the semantic gap between encoder and decoder sub-networks before fusion.
2. **Deep Supervision:** Auxiliary loss heads at nodes $X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4}$ force all intermediate feature maps to be discriminative.
3. **Model Pruning:** Enables runtime selection of pruned sub-networks (e.g., $L_1$ or $L_2$ for rapid clinical triage vs. $L_4$ for full-precision restorative treatment planning).
4. **Attention Gating:** Suppresses background alveolar bone patterns and highlights radiolucent cavities.

---

## 📐 Architecture & Mathematical Formulation

### Node Formulation
Let $x^{i,j}$ denote the output of node $X^{i,j}$, where $i$ indexes the downsampling level along the encoder ($0 \dots 4$) and $j$ indexes the convolution block along the dense skip pathway ($0 \dots 4$):

$$x^{i,j} = \begin{cases}
\mathcal{H}\left(x^{i-1, 0}\right), & j = 0 \\
\mathcal{H}\left( \left[ \left[x^{i,k}\right]_{k=0}^{j-1}, \, \mathcal{U}\left(x^{i+1, j-1}\right) \right] \right), & j > 0
\end{cases}$$

Where:
- $\mathcal{H}(\cdot)$ is a double convolution block: $\text{Conv}_{3\times3} \to \text{BatchNorm} \to \text{ReLU} \to \text{Conv}_{3\times3} \to \text{BatchNorm} \to \text{ReLU}$.
- $\mathcal{U}(\cdot)$ denotes bilinear upsampling by factor 2.
- $[\cdot]$ denotes concatenation along the channel axis.

```
Level 0:  X^{0,0} --------> X^{0,1} --------> X^{0,2} --------> X^{0,3} --------> X^{0,4} (Final Output)
             \             ^   \             ^   \             ^   \             ^
              v           /     v           /     v           /     v           /
Level 1:     X^{1,0} --------> X^{1,1} --------> X^{1,2} --------> X^{1,3}
                \             ^   \             ^   \             ^
                 v           /     v           /     v           /
Level 2:        X^{2,0} --------> X^{2,1} --------> X^{2,2}
                   \             ^   \             ^
                    v           /     v           /
Level 3:           X^{3,0} --------> X^{3,1}
                      \             ^
                       v           /
Level 4:              X^{4,0} (Bottleneck)
```

### Hybrid BCEDice Loss Function
To counter extreme pixel imbalance:

$$\mathcal{L}_{\text{total}} = \sum_{k=1}^{4} w_k \cdot \left[ \alpha \cdot \mathcal{L}_{\text{BCE}}(p^{(k)}, y) + (1 - \alpha) \cdot \mathcal{L}_{\text{Dice}}(p^{(k)}, y) \right]$$

Where:
- $\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i^2 + \sum y_i^2 + \epsilon}$
- $w = [0.1, 0.2, 0.3, 0.4]$ weights the progressive deep supervision outputs.

---

## 🏷️ Radiographic Caries Classification System

The built-in `LesionDetector` module performs connected-component morphological analysis on predicted masks and outputs ICDAS-equivalent radiographic severity grades:

| Grade | Radiographic Depth | Anatomical Description | Clinical Recommendation |
| :--- | :--- | :--- | :--- |
| **E1** | Outer Enamel | Radiolucency in outer 50% of enamel | Remineralization therapy (topical fluoride) |
| **E2** | Inner Enamel | Radiolucency in inner 50% up to EDJ | Non-invasive resin infiltration / monitoring |
| **D1** | Outer 1/3 Dentin | Demineralization extending past EDJ | Conservative composite resin restoration |
| **D2** | Middle 1/3 Dentin | Frank cavity in middle dentin | Restorative excavation and filling |
| **D3** | Deep Dentin / Pulp | Radiolucency within 0.5mm of pulp | High pulpal exposure risk; indirect pulp capping |

---

## 💻 Interactive Clinical Web Dashboard

The project includes an interactive, dark-mode web application:
- **Real-Time Sensitivity Control:** Dynamic threshold slider ($0.10$ to $0.90$) with instant live re-segmentation.
- **Multi-View Modes:**
  1. *Clinical Overlay:* Neon-tinted caries mask with colored bounding boxes and lesion tags.
  2. *Probability Heatmap:* Color gradient representing pixel cavity confidence.
  3. *CLAHE Enhanced View:* High-contrast equalization highlighting faint lucencies.
  4. *Side-by-Side Dual View:* Synchronized original vs. segmented radiograph.
- **Lesion Findings Table:** Itemized list of cavities with area (mm²), anatomical site, severity grade, and treatment advice.
- **Zero-Dependency Quickstart:** Built-in standalone engine runs out-of-the-box without waiting for large weights.

---

## 📁 Project Structure

```
dental-caries-detection-unetpp/
├── config.py                 # Central configuration (hyperparameters, paths)
├── requirements.txt          # Python dependencies
├── setup_env.bat             # Windows one-click setup script
├── setup_env.ps1             # PowerShell setup script
├── README.md                 # Complete documentation
│
├── models/                   # Deep Learning Models
│   ├── __init__.py
│   ├── unet_plus_plus.py     # Native PyTorch Nested U-Net with Deep Supervision & Attention
│   ├── losses.py             # BCEDiceLoss, FocalLoss, TverskyLoss, DeepSupervisionLoss
│   └── metrics.py            # Dice, IoU, Sensitivity, Specificity, Lesion F1
│
├── data/                     # Data Pipeline & Bootstrapping
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset, CLAHE preprocessing, DataLoaders
│   ├── augmentations.py      # Dental radiograph geometric & photometric augmentations
│   └── sample_generator.py   # Realistic synthetic dental radiograph & caries mask generator
│
├── training/                 # Model Training
│   ├── __init__.py
│   ├── trainer.py            # Trainer with AMP (mixed precision), early stopping & scheduler
│   └── train.py              # CLI entrypoint for model training
│
├── evaluation/               # Model Evaluation
│   ├── __init__.py
│   ├── evaluator.py          # Pixel & lesion-level evaluation + diagnostic comparison plates
│   └── evaluate.py           # CLI evaluation script
│
├── inference/                # Clinical Inference
│   ├── __init__.py
│   ├── predictor.py          # Full inference engine with TTA and diagnostic overlays
│   ├── lesion_detector.py    # Connected components, bounding boxes & severity grading
│   └── predict.py            # CLI prediction script for single images or directories
│
├── web_app/                  # Interactive Clinical Web Application
│   ├── app.py                # REST server (native http.server or Flask compatible)
│   ├── static/
│   │   ├── css/style.css     # Glassmorphic dark clinical design system
│   │   └── js/app.js         # Reactive UI logic, slider debouncing, report export
│   └── templates/
│       └── index.html        # Semantic HTML5 clinical workstation UI
│
├── tests/                    # Automated Test Suite
│   ├── test_model.py         # UNet++ forward shapes, deep supervision & pruning tests
│   ├── test_loss.py          # Loss functions & metric tracker tests
│   └── test_pipeline.py      # End-to-end integration tests
│
└── samples/                  # Preloaded demonstration dental radiographs
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Enter Project Directory
```powershell
cd C:\Users\MOHIT\.gemini\antigravity-ide\scratch\dental-caries-detection-unetpp
```

### 2. Install Dependencies
Run the automatic setup script:
```powershell
.\setup_env.ps1
```
Or manually using pip:
```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Web Dashboard
```powershell
python web_app/app.py
```
Open **[http://127.0.0.1:8501](http://127.0.0.1:8501)** in your browser to start testing radiographs immediately!

---

## 🏋️ Training Pipeline

### Generating Synthetic Radiographs & Training:
To train the UNet++ model immediately using synthetic bootstrapping:
```powershell
python training/train.py --epochs 40 --batch_size 8 --lr 0.001 --generate_sample --num_samples 64
```

### Training on Custom Clinical Datasets:
Place your dental X-rays and masks in `data/dataset/`:
```
data/dataset/
├── images/   # dental_001.png, dental_002.png, ...
└── masks/    # dental_001.png, dental_002.png, ...
```
Then run:
```powershell
python training/train.py --data_dir data/dataset --epochs 60 --batch_size 8
```

---

## 📊 Evaluation & Clinical Reporting

Evaluate a trained checkpoint against test radiographs:
```powershell
python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/dataset --threshold 0.50
```

This outputs:
- **Pixel-Level Metrics:** Dice Similarity Coefficient (DSC), IoU, Sensitivity, Specificity, Precision, Pixel Accuracy.
- **Lesion-Level Metrics:** Detected Lesion Count, Precision, Recall, Lesion F1 score.
- **Visual Diagnostic Comparison Plates:** 4-panel figures saved to `outputs/evaluation/visualizations/`.

---

## 🔍 Inference & Lesion Extraction

Run inference on a single dental X-ray:
```powershell
python inference/predict.py --input samples/sample_proximal.png --threshold 0.50 --output outputs/predictions
```

Run batch inference across an entire folder:
```powershell
python inference/predict.py --input data/dataset/images --output outputs/batch_predictions
```

Generates:
1. `*_annotated.png`: Radiograph with glowing coral caries overlays and labeled bounding boxes.
2. `*_heatmap.png`: High-resolution continuous probability distribution map.
3. `diagnostic_findings.json`: Itemized clinical findings, cavity areas, and treatment recommendations.

---

## 🧪 Unit & Integration Testing

Run the automated test suite:
```powershell
python -m unittest discover tests
```

---

## 📚 References
1. **Zhou, Z., et al.** (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* Deep Learning in Medical Image Analysis (DLMIA), LNCS 11045, pp. 3–11.
2. **Oktay, O., et al.** (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* arXiv:1804.03999.
3. **Schwendicke, F., et al.** (2020). *Artificial Intelligence in Dental Diagnostics: A Systematic Review.* Journal of Dental Research, 99(7), 769–776.

---

## ⚠️ Clinical Disclaimer
*This software is intended for research, academic, and decision-support demonstration purposes. It does not replace the professional diagnosis, clinical judgment, or treatment planning of a licensed dental practitioner.*
