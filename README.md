# EvidenceLens: 

Forensic evidence detection and crime classification from crime scene images.
<img width="2836" height="1226" alt="Screenshot 2026-05-07 184828" src="https://github.com/user-attachments/assets/b24a1dfc-dd66-4a0a-a243-93c07c0cfef9" />
<img width="2868" height="1576" alt="Screenshot 2026-05-07 184634" src="https://github.com/user-attachments/assets/a37f22e7-1d7c-4601-b17e-16abc0435130" />

https://huggingface.co/spaces/satyeaaa/EvidenceLens

EvidenceLens is a CPU-compatible computer vision system that detects 11 categories of forensic evidence and predicts crime type from a single crime scene image. It runs on standard hardware with no GPU required — 78ms per image at 12.8fps.
Built as a proof-of-concept for automated forensic triage, the system combines a YOLOv8-nano ensemble for evidence detection with a Random Forest classifier for crime type prediction.

# Results

ScoreEvidence Detection (mAP@0.5)56.6%Crime Classification Accuracy98.2% ± 1.1%Inference Speed78ms / image (CPU)Dataset1,050 images, 6,847 annotations
The ensemble architecture improved overall mAP by 9.7% over a single model baseline. The largest gains were on small trace evidence — prints (+99.1%), gloves (+78.5%), shell casings (+53.0%) — which a single model consistently missed.
The 56.6% mAP reflects real constraints: fingerprints at 15-25 pixels fall below YOLOv8's optimal detection range, and the 1,050 image dataset is small for 11-class detection. The next meaningful improvement would come from dataset expansion to 3,000+ images and attention mechanisms (CBAM) for small object sensitivity.

# What It Does

Detects 11 evidence classes: blood, gun, knife, bullet hole, shell casing, fingerprint, gloves, prints, crime scene tape, drink can, background
Predicts crime type across 5 categories: homicide, armed robbery, burglary, assault, vandalism
Dual-model ensemble with soft NMS fusion (IoU 0.6) for improved trace evidence detection
Streamlit dashboard with adjustable confidence threshold, export to PNG/JSON/CSV


# Tech Stack
YOLOv8 scikit-learn Streamlit OpenCV PyTorch Python


## System Architecture

### Evidence Detection (YOLOv8-Nano Ensemble)

**11 Evidence Classes:**
- Biological: Blood
- Weapons: Gun, Knife
- Ballistics: Bullet Hole, Shell Casing
- Trace: Fingerprint, Gloves, Prints (shoeprints/footprints)
- Context: Crime Scene Tape, Drink Can, Background

**Dual Model Ensemble:**
- **Model V3:** Primary detector covering all 11 classes (weapons, ballistics, 
  biological, trace, context). Trained for 125 epochs, achieves 57.9% mAP@0.5 
  at epoch 48 (best checkpoint)
- **Model V2:** Specialist for prints detection (shoeprints, fingerprints). 
  Provides enhanced small-object sensitivity for trace evidence
- **Fusion:** Soft NMS (IoU 0.6) merges detections, keeping highest-confidence 
  predictions when models overlap. App gracefully handles missing V2 model

### Crime Classification (Random Forest)

**5 Crime Types:**
- Homicide
- Armed Robbery
- Burglary
- Assault
- Vandalism

**Configuration:**
- 200 trees, maximum depth 10
- Class-balanced weights
- 10 evidence count features (background excluded)
- 5-fold cross-validation: 98.2% ± 1.1% accuracy

---

## Repository Structure

```
EvidenceLens/
├── README.md                              # This file
├── requirements.txt                        # Python dependencies
├── app.py                                 # Streamlit dashboard (main application)
│
├── models/                                # Trained models
│   ├── evidencelens_v3.pt                 # Main YOLO model (weapons, ballistics, blood)
│   ├── evidencelens_v2.pt                 # Prints specialist model (shoeprints, fingerprints)
│   ├── crime_classifier_py311.pkl         # Random Forest crime classifier
│   └── feature_info_py311.pkl             # Feature metadata for classifier
│
├── data/                                  # Dataset and labels
│   ├── train/                             # Training images and labels
│   ├── val/                               # Validation images and labels
│   ├── raw/                               # Raw unprocessed data
│   ├── dataset.yaml                       # YOLO dataset configuration
│   ├── evidence_features.csv              # Extracted evidence counts
│   ├── crime_labels.csv                   # Crime type annotations
│   └── crime_labels_suggested.csv         # Classifier predictions
│
├── runs/                                  # Training outputs
│   └── detect/
│       └── evidencelens_final_v3/         # V3 training run
│           ├── weights/
│           │   ├── best.pt                # Best checkpoint
│           │   └── last.pt                # Final checkpoint
│           ├── results.csv                # Training metrics
│           ├── results.png                # Training curves
│           └── confusion_matrix.png       # Validation confusion matrix
│
└── src/                                   # Utility scripts
    ├── extract_evidence_features.py       # Feature extraction for classifier
    ├── train_crime_classifier.py          # Random Forest training
    └── compute_rf_accuracy.py             # Accuracy computation
```

---

## Installation & Setup

### System Requirements

- **Operating System:** Windows 10/11, Linux (Ubuntu 22.04+), macOS
- **Python:** 3.8-3.10
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 5GB for models and dependencies
- **Hardware:** CPU-only compatible (tested on ARM Snapdragon X Elite)

### Installation Steps

1. **Clone or extract the project:**
```bash
cd EvidenceLens/
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import ultralytics; import streamlit; import joblib; print('✓ All dependencies installed')"
```

---

## Usage

### Option 1: Streamlit Dashboard (Recommended)

**Launch interactive web interface:**
```bash
streamlit run app.py
```

**Required files (app checks these paths):**
- `models/evidencelens_v3.pt` - Main YOLO model (required)
- `models/evidencelens_v2.pt` - Prints specialist (optional, app works without it)
- `models/crime_classifier_py311.pkl` - Random Forest classifier (optional)
- `models/feature_info_py311.pkl` - Feature metadata (optional)

**Features:**
- Upload crime scene images (PNG, JPG, JPEG)
- Dual-model ensemble detection with soft NMS fusion
- Adjustable confidence threshold (0.1-1.0, default 0.25)
- Real-time detection visualization with color-coded categories
- Evidence count summary by forensic category
- Crime type prediction with confidence scores
- Export options: PNG, JSON, CSV
- Recent detections gallery with thumbnails

**Default URL:** http://localhost:8501

---

## Training Your Own Models

### Dataset Preparation

1. **Organize images:**
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

2. **Create dataset.yaml:**
```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: background
  1: blood
  2: bullet_hole
  3: crime_scene_tape
  4: drink_can
  5: fingerprint
  6: gloves
  7: gun
  8: knife
  9: shell_casing
  10: prints
```

### Train YOLO Model

```bash
python train_final_fixed.py
```

**Key parameters (edit in script):**
- Epochs: 100
- Batch size: 16
- Image size: 640×640
- Optimizer: AdamW
- Learning rate: 0.01 (cosine decay to 0.0001)

### Train Crime Classifier

```bash
# Extract evidence features from detections
python extract_evidence_features.py

# Train Random Forest
python train_crime_classifier.py
```

---

## Performance Metrics

### Detection Performance (Ensemble)

| Evidence Class    | Baseline | Ensemble | Improvement |
|-------------------|----------|----------|-------------|
| Blood             | 88.4%    | 87.9%    | -0.6%       |
| Gun               | 65.8%    | 69.0%    | +4.9%       |
| Knife             | 68.9%    | 71.4%    | +3.6%       |
| Bullet Hole       | 65.2%    | 66.1%    | +0.9%       |
| Shell Casing      | 43.8%    | 67.0%    | +53.0%      |
| Fingerprint       | 27.3%    | 41.1%    | +50.5%      |
| Gloves            | 30.2%    | 53.9%    | +78.5%      |
| Prints            | 21.2%    | 42.2%    | +99.1%      |
| Crime Tape        | 73.5%    | 74.8%    | +1.3%       |
| Drink Can         | 42.1%    | 45.3%    | +3.2%       |
| Background        | 95.8%    | 95.2%    | -0.6%       |
| **Overall mAP**   | **51.6%**| **56.6%**| **+9.7%**   |

### Crime Classification Performance

| Crime Type      | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Homicide        | 0.98      | 0.99   | 0.98     | 112     |
| Armed Robbery   | 0.97      | 0.97   | 0.97     | 89      |
| Burglary        | 0.99      | 0.99   | 0.99     | 97      |
| Assault         | 0.99      | 0.99   | 0.99     | 78      |
| Vandalism       | 1.00      | 1.00   | 1.00     | 47      |
| **Accuracy**    |           |        | **98.2%**| **423** |

---

## Known Limitations

### Dataset Constraints
- Small scale: 1,050 images (recommended 3,000-5,000+ for production)
- Limited diversity: ~25% staged scenes, underrepresented outdoor/night conditions
- Single annotator: Potential subjective bias in labeling

### Technical Limitations
- Small object detection: Fingerprints (15-25 pixels) below YOLOv8 optimal range (≥32×32)
- No spatial reasoning: Crime classifier ignores evidence proximity/relationships
- CPU inference speed: 78ms suitable for triage, insufficient for real-time video
- Confidence calibration: Scores not calibrated to reflect actual reliability

### Deployment Constraints
- No regulatory validation: Not approved by UK Forensic Science Regulator
- Proof-of-concept only: Requires extensive testing for operational use
- Generalization limits: Performance may degrade outside training distribution

---

## Future Work

### High Priority
1. **Dataset Expansion:** 3,000-5,000 images with environmental diversity
2. **Explainability:** Grad-CAM visualization, confidence calibration
3. **Spatial Features:** Evidence proximity, dispersion patterns for classification

### Medium Priority
4. **Attention Mechanisms:** CBAM, Coordinate Attention for small objects
5. **Multi-scale Training:** Variable input sizes (512-768 pixels)
6. **Learned Fusion:** Adaptive V2/V3 weighting instead of fixed NMS

### Low Priority
7. **Real-time Video:** Temporal tracking for surveillance footage
8. **Multi-label Classification:** Handle mixed-motive crime scenes
9. **Cross-dataset Validation:** Test on international crime scene datasets

---

## Citation

If you use this work, please cite:

```
SATYA (2025). EvidenceLens: Computer Vision for Forensic Evidence Detection 
and Crime Classification. BSc Computer Science Final Year Project, 
Anglia Ruskin University.
```

## License & Ethics

### License
This project is submitted as academic coursework for Anglia Ruskin University and is not licensed for commercial use without permission.

### Ethical Considerations
- All data sourced from publicly available datasets (Kaggle, validated repositories)
- No identifiable individuals, private property, or sensitive crime scene information
- Staged evidence and forensic training materials used
- Ethics approval: ETH2526-1936 (LOW RISK)
- Compliant with UK Data Protection regulations

### Intended Use
**FOR:** Academic research, forensic training, proof-of-concept demonstrations  
**NOT FOR:** Operational forensic deployment without regulatory validation

---

## Acknowledgments

- **YOLOv8:** Ultralytics team for open-source object detection framework
- **Datasets:** Kaggle contributors, Roboflow community
- **Libraries:** Streamlit, scikit-learn, OpenCV, PyTorch

---

## Version History

- **v1.0 (April 2025):** Initial submission version
  - Ensemble YOLOv8-nano detection (56.6% mAP)
  - Random Forest crime classification (98.2% accuracy)
  - CPU-only training and deployment
  - Streamlit dashboard interface

---

**Last Updated:** December 2025  
**Project Status:** Complete 
