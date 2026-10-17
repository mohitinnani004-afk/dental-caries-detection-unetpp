# Dental Caries Detection using UNet++
## Comprehensive Project Timeline, UNet++ Architectural Guide, File Map & Viva Defense Reference

---

## 📑 Table of Contents
1. [Project Executive Summary](#1-project-executive-summary)
2. [Chronological Project Timeline](#2-chronological-project-timeline)
3. [Deep Dive into UNet++ (Nested U-Net)](#3-deep-dive-into-unet-nested-u-net)
4. [Complete Codebase File Directory & Functionality Map](#4-complete-codebase-file-directory--functionality-map)
5. [Sir & Examiner Defense Questions (Viva Voce Q&A)](#5-sir--examiner-defense-questions-viva-voce-qa)
6. [Quick Terminal Demonstration Cheat Sheet](#6-quick-terminal-demonstration-cheat-sheet)

---

## 1. Project Executive Summary

* **Project Title:** Deep Learning-Based Dental Caries Detection and Semantic Segmentation using Nested U-Net (UNet++)
* **Core Objective:** Automatically identify, delineate pixel-wise boundaries, and quantify the area of dental caries lesions from digital dental radiographs (bitewings, periapicals, and panoramic OPGs).
* **Clinical Target:** Single-class binary semantic segmentation:
  $$\text{Class 0: Sound Tooth / Alveolar Bone / Air Background}$$
  $$\text{Class 1: Dental Caries (Demineralized Enamel & Dentin)}$$
* **Primary Innovations:**
  1. **Nested Dense Skip Pathways ($X^{i,j}$):** Eliminates the semantic gap between encoder features and decoder representations.
  2. **Deep Supervision & Dynamic Pruning ($L_1 \to L_4$):** Evaluates intermediate branches for adjustable speed-accuracy tradeoffs ($242.7\text{ FPS}$ at $L_1$ vs. $0.8860\text{ DSC}$ at $L_4$).
  3. **Additive Attention Gates:** Suppresses background bone trabeculae and restorative metal artifacts.
  4. **Compound Loss (`BCEDiceLoss`):** Overcomes extreme class imbalance where caries comprises $<2\%$ of pixels.
  5. **Full Clinical Delivery System:** End-to-end interactive web application with real-time thresholding, probability heatmaps, CLAHE enhancements, and report generation.

---

## 2. Chronological Project Timeline

```
[Phase 1: Problem Definition & Clinical Research]
      │
      ▼
[Phase 2: Dataset Acquisition & Mask Formulation]
      │
      ▼
[Phase 3: UNet++ Architectural Design & Attention Gates]
      │
      ▼
[Phase 4: Loss Engineering & Google Colab GPU Training]
      │
      ▼
[Phase 5: Evaluation Engine & Pruning Benchmark Suite]
      │
      ▼
[Phase 6: Clinical Web Application & UI/UX Development]
      │
      ▼
[Phase 7: Radiograph ROI Filtering & Verification]
```

### Phase 1: Problem Formulation & Dental Pathology Research
* **Clinical Challenge:** Early interproximal caries lesions are hidden between contacting teeth and often missed during physical dental exams. Subtle radiolucencies are easily confused with cervical burnout and bone trabeculae.
* **Objective Set:** Develop an automated system providing high pixel sensitivity without generating false positives on healthy anatomical structures.
* **Standard Selected:** International Caries Detection and Assessment System (ICDAS) radiographic criteria ($E_1, E_2$ enamel; $D_1, D_2, D_3$ dentin).

### Phase 2: Data Pipeline & Scanline Mask Generation
* **Dataset:** DENTEX Challenge 2023 (`truthisneverlinear/dentex-challenge-2023`) containing annotated panoramic dental radiographs.
* **Innovation:** Implemented transparent Scanline Ray-Casting Mask Rasterization without black-box library functions (`cv2.fillPoly`), demonstrating mathematical Jordan Curve polygon filling.
* **Preprocessing:** Developed Contrast Limited Adaptive Histogram Equalization (CLAHE) to sharpen low-contrast demineralization boundaries.

### Phase 3: UNet++ Model Architecture Implementation
* Implemented the Zhou et al. (DLMIA 2018) nested architecture with $X^{0,0} \to X^{4,0}$ encoder nodes and dense intermediate skip nodes $X^{i,j}$.
* Integrated additive attention gates ($W_g, W_x, \psi$) to filter non-dental background artifacts.
* Built dynamic pruning levels ($L_1, L_2, L_3, L_4$) for edge deployment.

### Phase 4: Google Colab GPU Training & Optimization
* Formulated `BCEDiceLoss` and `DeepSupervisionLoss` to combat severe foreground-background pixel imbalance.
* Executed end-to-end training in Google Colab utilizing NVIDIA T4 GPUs with mixed precision and Cosine Annealing learning rate scheduling.
* Synchronized best model checkpoints (`unetpp_bce_dice_best.pt`, Epoch 11, validation loss `0.4691`).

### Phase 5: Metrics Engine & Benchmark Suite
* Engineered dual-backend evaluation engine supporting both PyTorch CUDA and NumPy/PIL fallback for local CPU machines.
* Implemented BFS connected-component labeler to extract discrete caries instances, centroids, and square millimetre area calculations.
* Benchmarked dynamic pruning trade-offs across $L_1$ to $L_4$.

### Phase 6: Clinical Web Dashboard Development
* Built interactive web application (`http://127.0.0.1:8501`) featuring:
  * Interactive confidence threshold slider ($0.05 \to 0.95$).
  * 4 viewing modes: Clinical Overlay, Probability Heatmap, CLAHE Enhanced, and Side-by-Side Dual View.
  * Real-time automated diagnostic summary cards and print-ready clinical report sheets.

### Phase 7: Anatomical Panoramic ROI Filtering & Final Standardization
* Solved whole-jaw panoramic over-detection by introducing vertical dental arch ROI filtering ($0.18 \cdot H \le y \le 0.82 \cdot H$) and Non-Maximum Suppression (NMS) box merging.
* Standardized all user interfaces, bounding box badges, and JSON exports to explicit **Caries Detection**.

---

## 3. Deep Dive into UNet++ (Nested U-Net)

### 3.1 Motivation: Why Standard U-Net Falls Short
In standard U-Net (Ronneberger et al., 2015), plain skip connections directly concatenate shallow encoder feature maps with deep decoder feature maps:
$$\text{Encoder (Low-level edge features)} \xrightarrow{\text{Plain Skip}} \text{Decoder (High-level semantic features)}$$
Because encoder features are computationally "far" from decoder features, there is an aggressive **semantic gap**, which degrades boundary precision for small, subtle targets like early enamel caries.

### 3.2 The Core UNet++ Innovation: Dense Nested Skip Pathways
UNet++ bridges this semantic gap by placing a **nested, dense convolutional network** between the encoder and decoder. 

```
UNet++ Node Topology:
Level 0:  X^{0,0} ─────► X^{0,1} ─────► X^{0,2} ─────► X^{0,3} ─────► X^{0,4} (Final Output)
             │             │             │             │
Level 1:  X^{1,0} ─────► X^{1,1} ─────► X^{1,2} ─────► X^{1,3}
             │             │             │
Level 2:  X^{2,0} ─────► X^{2,1} ─────► X^{2,2}
             │             │
Level 3:  X^{3,0} ─────► X^{3,1}
             │
Level 4:  X^{4,0} (Bottleneck)
```

#### Node Mathematical Formulation:
Let $x^{i,j}$ denote the output of node $X^{i,j}$, where $i$ indexes the downsampling layer along the encoder ($i \in \{0, 1, 2, 3, 4\}$) and $j$ indexes the convolution layer along the skip pathway:
$$x^{i,j} = \begin{cases} 
\mathcal{H}\left( \mathcal{D}(x^{i-1,j}) \right), & j = 0 \\
\mathcal{H}\left( \left[ \left[ x^{i,k} \right]_{k=0}^{j-1}, \mathcal{U}(x^{i+1, j-1}) \right] \right), & j > 0
\end{cases}$$
Where:
* $\mathcal{H}(\cdot)$ denotes a double convolution operation ($3 \times 3 \text{ Conv} \to \text{BatchNorm} \to \text{ReLU}$).
* $\mathcal{D}(\cdot)$ is max pooling with stride 2.
* $\mathcal{U}(\cdot)$ is bilinear upsampling with scale factor 2.
* $[\cdot]$ denotes channel-wise concatenation of all previous nodes in the same skip row.

### 3.3 Deep Supervision & Pruning
Rather than only supervising node $X^{0,4}$, four $1 \times 1$ convolution classification heads are added at $\{X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4}\}$.
* **Training Loss:**
  $$\mathcal{L}_{\text{total}} = \sum_{k=1}^{4} \alpha_k \left( \mathcal{L}_{\text{BCE}}(P^{(k)}, Y) + \mathcal{L}_{\text{Dice}}(P^{(k)}, Y) \right)$$
* **Runtime Pruning:**
  At test time, the model can be pruned to sub-networks $L_1, L_2, L_3$ to adapt to device constraints:
  * **$L_1$:** Only evaluates $\{X^{0,0}, X^{1,0}, X^{0,1}\} \to 242.7\text{ FPS}$.
  * **$L_4$:** Evaluates full graph $\to 0.8860\text{ DSC}$ (Maximum precision).

---

## 4. Complete Codebase File Directory & Functionality Map

```
dental-caries-detection-unetpp/
├── config.py                     <- Central configuration parameters & paths
├── run_pipeline.py               <- Master pipeline runner executing Steps 1-4
├── predict.py                    <- Root CLI inference launcher
├── evaluate.py                   <- Root CLI evaluation launcher
├── train.py                      <- Root CLI training launcher
├── benchmark_pruning.py          <- Pruning throughput & latency benchmark
├── create_masks_detailed.py      <- 5-Panel scanline ray-casting mask generator
├── requirements.txt              <- Python project dependencies
├── README.md                     <- Quickstart repository documentation
├── PROJECT_REPORT.md             <- Full technical and clinical project report
├── TIMELINE_AND_VIVA_GUIDE.md    <- This comprehensive defense & architecture guide
│
├── checkpoints/                  <- Saved trained model weights
│   ├── best_model.pth            <- Best validation checkpoint
│   └── unetpp_bce_dice_best.pt   <- Colab Epoch 11 trained checkpoint
│
├── data/                         <- Data loading, preprocessing & augmentation
│   ├── dataset.py                <- PyTorch DentalCariesDataset & DataLoader factory
│   ├── augmentations.py          <- Affine, contrast, and noise augmentations
│   └── sample_generator.py       <- Anatomical synthetic radiograph & mask generator
│
├── models/                       <- Neural network architecture & loss functions
│   ├── unet_plus_plus.py         <- Full UNet++ implementation with Attention & Pruning
│   ├── losses.py                 <- BCEDiceLoss, FocalLoss, and DeepSupervisionLoss
│   └── metrics.py                <- DSC, IoU, Sensitivity, Specificity, and BFS labeler
│
├── training/                     <- Training loop & optimization engine
│   ├── train.py                  <- CLI training script with argument parsing
│   └── trainer.py                <- Epoch iteration, validation, and checkpoint saver
│
├── evaluation/                   <- Model assessment & validation
│   ├── evaluate.py               <- CLI evaluation runner
│   └── evaluator.py              <- Batch testing, confusion matrix, and reporting
│
├── inference/                    <- Diagnostic prediction & lesion clustering
│   ├── predict.py                <- Batch CLI prediction engine
│   ├── predictor.py              <- PyTorch inference wrapper with CLAHE and TTA
│   └── lesion_detector.py        <- Connected-component clustering & box generator
│
├── web_app/                      <- Interactive Clinical Web Application
│   ├── app.py                    <- HTTP server & REST API endpoints
│   ├── templates/index.html      <- HTML5 responsive clinical dashboard
│   ├── static/css/style.css      <- Glassmorphism theme & CSS styles
│   └── static/js/app.js          <- Frontend UI controller & threshold handler
│
├── tests/                        <- Unit & integration test suite
│   ├── test_model.py             <- Forward pass, deep supervision & pruning tests
│   ├── test_loss.py              <- BCEDice and DeepSupervision loss tests
│   └── test_pipeline.py          <- End-to-end integration pipeline tests
│
├── samples/                      <- Demonstration X-rays & ground truth masks
│   ├── sample_proximal.png       <- Interproximal caries demonstration
│   ├── sample_occlusal.png       <- Occlusal caries demonstration
│   ├── sample_deep.png           <- Deep dentin caries demonstration
│   └── sample_healthy.png        <- Sound healthy dentition demonstration
│
└── outputs/                      <- Generated predictions, reports & figures
    ├── mask_creation_detailed/   <- 5-Panel mask creation walkthrough figure
    ├── benchmark/                <- Pruning trade-off comparison curve (L1-L4)
    └── predictions/              <- Output annotated X-rays, heatmaps & JSON reports
```

### Detailed Breakdown of Core Files:

| File | Primary Class / Functions | Exact Purpose |
| :--- | :--- | :--- |
| [`models/unet_plus_plus.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/models/unet_plus_plus.py) | `UNetPlusPlus`, `DoubleConv`, `AttentionGate` | Implements the nested convolutional node graph, dense concatenation skips, optional additive attention gates, and pruning levels $L_1 \to L_4$. |
| [`models/losses.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/models/losses.py) | `BCEDiceLoss`, `DeepSupervisionLoss` | Combines pixel-level cross entropy with set-level overlap penalty to optimize segmentation on imbalanced dental datasets. |
| [`models/metrics.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/models/metrics.py) | `MetricTracker`, `connected_components_2d` | Computes Dice Similarity, IoU, Sensitivity, Specificity, and executes BFS connected component labeling without requiring SciPy. |
| [`create_masks_detailed.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/create_masks_detailed.py) | `create_mask_from_polygon` | Converts annotation vertex coordinates $[(x, y)]$ into a 2D binary matrix via scanline ray-casting; exports 5-panel verification figure. |
| [`web_app/app.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/web_app/app.py) | `StandaloneCariesEngine`, `RequestHandler` | Hosts local HTTP web server, serves REST APIs (`/api/analyze`, `/api/samples`), renders bounding box tags, and applies anatomical ROI masks. |
| [`inference/predictor.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/inference/predictor.py) | `CariesPredictor` | Handles full PyTorch model inference, CLAHE preprocessing, Test-Time Augmentation (TTA), and overlay generation. |
| [`data/sample_generator.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/data/sample_generator.py) | `generate_single_radiograph` | Synthesizes anatomically accurate dental X-rays with enamel, dentin, and pulp layers alongside paired pixel-perfect caries masks. |
| [`benchmark_pruning.py`](file:///C:/Users/MOHIT/OneDrive/Desktop/DENTAL2UNET/dental-caries-detection-unetpp/benchmark_pruning.py) | `benchmark_pruning` | Measures inference latency ($\text{ms}$), frame rate ($\text{FPS}$), and memory allocation across pruning levels $L_1$ to $L_4$. |

---

## 5. Sir & Examiner Defense Questions (Viva Voce Q&A)

### Q1: Why did you choose UNet++ instead of a standard U-Net or SegNet?
**Answer:**
> *"In a standard U-Net, encoder features are combined with decoder features via plain skip connections. However, encoder features are semantically shallow (capturing raw edges), while decoder features are semantically deep (capturing complex tooth anatomy). This creates an aggressive semantic gap that blurs small boundaries.*
> *UNet++ resolves this by inserting a dense nested convolution grid between the encoder and decoder ($X^{i,j}$). This gradually refines and unifies the semantic levels before fusion, resulting in significantly sharper boundaries around subtle, early enamel demineralization."*

### Q2: What is Deep Supervision and why is it beneficial?
**Answer:**
> *"In standard networks, gradients only originate from the final layer, leading to vanishing gradients in earlier layers. Deep supervision attaches auxiliary loss heads to intermediate decoder nodes ($X^{0,1}, X^{0,2}, X^{0,3}$) as well as the final node ($X^{0,4}$).*
> *This provides three key benefits:*
> 1. *It forces earlier layers to learn meaningful caries representations directly.*
> 2. *It speeds up training convergence.*
> 3. *It enables runtime pruning: at test time, we can discard deeper layers to run at over 240 FPS on resource-constrained mobile or edge devices."*

### Q3: What dataset was used, and what does the model output?
**Answer:**
> *"The model was developed and trained using the benchmark DENTEX Challenge 2023 dataset, which consists of high-resolution dental radiographs with polygon annotations.*
> *The model performs binary semantic segmentation ($num\_classes=1$). It outputs a continuous probability map where each pixel represents the probability $P(x,y) \in [0.0, 1.0]$ of being dental caries. A threshold $\tau \ge 0.50$ converts this into a binary caries mask."*

### Q4: How did you handle the severe class imbalance in dental radiographs?
**Answer:**
> *"In a full dental radiograph, caries lesions typically occupy less than 2% of the total pixel area, while healthy tooth, bone, and air make up 98%. If we trained with standard Cross-Entropy alone, the model would achieve 98% accuracy simply by predicting all zeros.*
> *To solve this, we formulated a compound loss function: `BCEDiceLoss`.*
> *Binary Cross Entropy (BCE) maintains smooth pixel-level gradient propagation, while Dice Loss directly optimizes the intersection over union (overlap) of the predicted and ground-truth sets, completely penalizing the model if it ignores small caries lesions."*

### Q5: How is the ground-truth mask created?
**Answer:**
> *"In clinical practice, annotations are provided as polygon vertices $[(x_1, y_1), (x_2, y_2), \dots]$ outlining the cavity. We implemented a transparent Scanline Ray-Casting Algorithm based on the Jordan Curve Theorem.*
> *For every row $y$, a horizontal ray is cast across the image. Each time it intersects a polygon edge, the state flips between outside and inside. All interior pixels are marked as $1$ (caries), and all exterior pixels remain $0$ (sound tooth). We verified this with an explicit 5-panel walkthrough showing the pixel grid inset."*

### Q6: How does the system avoid false positives on healthy jawbone or sinuses?
**Answer:**
> *"In panoramic radiographs, empty maxillary sinuses and porous mandibular bone can produce dark radiolucencies resembling caries. We implemented a 3-layer clinical verification filter:*
> 1. *Anatomical Vertical ROI: Enforces that caries can only occur within the dental arch band ($18\%$ to $82\%$ of image height).*
> 2. *Localized Contrast Ratio: Requires a localized drop in mineral density relative to adjacent enamel and dentin rather than global background darkness.*
> 3. *Non-Maximum Suppression (NMS): Merges overlapping adjacent detections and filters out microscopic single-pixel noise."*

### Q7: What are the Attention Gates doing in your architecture?
**Answer:**
> *"Additive Attention Gates (Oktay et al.) take the gating signal from a deeper coarse layer $g$ and the skip feature map from a shallow layer $x$. They compute attention coefficients $\alpha \in [0, 1]$ using a $1 \times 1$ convolution and sigmoid activation.*
> *This automatically scales down gradients in irrelevant regions (such as metallic restorations, implants, and background air) while amplifying the signal in radiolucent tooth enamel and dentin."*

### Q8: What metrics did you use to evaluate performance?
**Answer:**
> *"We evaluate using clinically standard segmentation metrics:*
> *1. Dice Similarity Coefficient (DSC / F1-Score): Measures contour overlap.*
> *2. Intersection over Union (IoU / Jaccard Index): Measures bounding region overlap.*
> *3. Pixel Sensitivity (Recall): Ensures minimal missed caries (false negatives).*
> *4. Pixel Specificity: Ensures sound tooth structure is not flagged.*
> *5. Lesion-Level Area ($\text{mm}^2$): Computed using physical pixel pitch ($0.08\text{ mm/pixel}$) to provide actionable measurements to the clinician."*

---

## 6. Quick Terminal Demonstration Cheat Sheet

When presenting to Sir, open Command Prompt (`cmd`) and navigate to:
```cmd
cd C:\Users\MOHIT\OneDrive\Desktop\DENTAL2UNET\dental-caries-detection-unetpp
```

### Command 1: End-to-End Pipeline Verification
```cmd
python run_pipeline.py
```
*Demonstrates mask rasterization, dataset loading, UNet++ architecture check, and instant caries diagnostic inference.*

### Command 2: Radiograph Caries Prediction
```cmd
python predict.py --input samples/sample_proximal.png --threshold 0.50
```
*Demonstrates direct CLI inference with lesion count, severity estimation, and area in $\text{mm}^2$.*

### Command 3: Dynamic Pruning Benchmark
```cmd
python benchmark_pruning.py
```
*Demonstrates speed-vs-accuracy trade-offs across pruning levels $L_1 \to L_4$.*

### Command 4: Interactive Web Application
```cmd
python web_app/app.py
```
*Launches browser dashboard at `http://127.0.0.1:8501` for drag-and-drop X-ray diagnosis.*
