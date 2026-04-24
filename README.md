# ❯ DermaAI: Multi-Model Dermatology Disease Detection System

<div align="center">

Built with the tools and technologies:

<img src="https://img.shields.io/badge/Keras-D00000.svg">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg">
<img src="https://img.shields.io/badge/NumPy-013243.svg">
<img src="https://img.shields.io/badge/pandas-150458.svg">
<img src="https://img.shields.io/badge/OpenCV-27338e.svg">
<img src="https://img.shields.io/badge/Albumentations-4B8BBE.svg">

</div>

---

## Overview

DermaEnsemble AI is a comprehensive dermatology disease detection system designed to solve one of the most critical problems in tele-dermatology: **reliable and clinically meaningful diagnosis from Web Apps/smartphone-captured images**.

The system uses a **multi-layered, multi-model architecture** combining:

- **Image validation techniques**
- **Binary filtering**
- **Deep learning classifiers**
- Uses a single powerful model:
        **EfficientNet-V2M**
    Produces final softmax outputs
    Provides stable and accurate predictions
    Handles class imbalance effectively

This ensures the system can handle real-world noisy, blurry, or irrelevant images and still deliver high-quality predictions across **seven dermatology categories**.


---

## Why This System?

### Solves These Key Problems

1. **Low-quality smartphone images**  
   → System validates blur, corruption, and image quality.

2. **Irrelevant (non-skin) submissions**  
   → MobileNetV3-Large binary classifier removes non-skin inputs.

3. **Complex dermatology classification**  
   → EfficientNet-V2M provides strong accuracy without multi-model ensembles.

---

## Problem Solve 
Dermatology tele-consultations rely heavily on user-submitted images, which are often corrupted, blurred, or contain irrelevant content, leading to misdiagnosis.

This project addresses the real-world challenge of automated dermatology triage by implementing a multi-stage pipeline consisting of image corruption detection, clinical-grade sharpness assessment, skin detection (binary classification), and a robust 7-class dermatology classification model.

The system improves diagnostic reliability and reduces error by ensuring that only valid, high-quality skin images are forwarded to the classifier.

---

## Techniques Used

### 1. Image Integrity & Blur Detection
- Laplacian blur scoring  
- Noise-tolerant blur threshold  
- Corruption detection with PIL + OpenCV  

### 2. Skin / Non-Skin Binary Filtering
- Lightweight MobileNetV3-Large model  
- Ensures only dermatology images reach the classifier  

### 3. 7-Class Dermatology Classification  
The EfficientNet-V2M model predicts:

1. Acne  
2. Eczema / Dermatitis  
3. Fungal infections  
4. Pigmentation disorders  
5. Benign lesions  
6. Viral/Bacterial infections  
7. Other dermatological abnormalities  

---

## Prediction Strategy

### Uses a Single Strong Model  
**EfficientNet-V2M**

### Benefits  
- Produces stable and accurate softmax outputs  
- Faster inference  
- Lower memory usage  
- Handles class imbalance effectively  
- Easier deployment across CPU & GPU  

---

## Architecture Diagram
<img src="Architecture/Model, Architc-2025-11-18-194137.svg" width="100%" style="max-height:80px; object-fit:contain;">

---

## Pipeline Summary

### Layer 1 — Image Validation  
Reject blurry, corrupted, or invalid images  

### Layer 2 — Binary Skin Detector  
MobileNetV3-Large (98.6% accuracy)  

### Layer 3 — Final Disease Classification  
EfficientNet-V2M (90% accuracy)

---

## Model Architecture (Layer-by-Layer)

```
Incoming Image
   │
   ├── Layer 1: Image Validity + Blur Detection
   │
   ├── Layer 2: Binary Model (Skin / Non-Skin)
   │
   └── Layer 3: EfficientNet-V2M (7-Class Classification)
```

---

## Model Performance

### Binary Classifier (Skin vs Non-Skin)
| Model | Accuracy |
|-------|---------:|
| MobileNetV3-Large | **98.6%** |
| EfficientNet-B0 | **97%** |

### 7-Class Dermatology Classifier
| Model | Accuracy |
|-------|---------:|
| EfficientNet-V2M | **90%** |
| EfficientNet-B4 | **88%** |
| MobileNetV3-Large | **86%** |

---

## Why These Models Were Chosen?

### **EfficientNet-V2M**
- Excellent balance of speed and accuracy  
- Top performer (90% accuracy)  
- Handles texture-based features very well  

### **EfficientNet-B4**
- Good for high-resolution medical images  
- Larger receptive field  
- Strong performance on skin lesion patterns  

### **MobileNetV3-Large**
- Lightweight & optimized for mobile/edge  
- Fast inference (~10–20ms on CPU)  
- Complements EfficientNet models in ensemble diversity  

### **MobileNetV3-Large (Binary Filter)**
- High confidence in classifying skin vs non-skin  
- Works extremely well for filtering irrelevant content  

---

## Project Structure

```
project/
│── app/
│── data/
│── docs/
│── reports/
│── models/
│   ├── 7_class_model/
│   └── Binary model/
│── Architecture/
│── notebooks/
│── requirements.txt
│── README.md
```

---

## Installation

```bash
git clone <repo-url>
cd <project>
pip install -r requirements.txt
```

---

## Usage Example

```bash
python app/run_inference.py --image path/to/image.jpg
```

---

## License
MIT License.

---

## Acknowledgments
Thanks to the open‑source community and dataset providers.
