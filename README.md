# Multilabel Chest X-ray Classification Using Swin Transformer & EfficientNet-B0

This repository contains the complete code, processing pipeline, model training scripts, and evaluation tools for **multilabel chest X-ray (CXR) classification** using:

- **Swin Transformer (Swin-Tiny)**
- **EfficientNet-B0**

The goal is to classify five CheXpert findings:

- Cardiomegaly  
- Pneumonia  
- Atelectasis  
- Pleural Effusion  
- No Finding  

The project includes a patient-wise evaluation protocol, statistical significance testing, and interpretability (Grad-CAM).

## 📂 Repository Structure

├── src/
│   ├── dataset.py            # Dataset loading & preprocessing
│   ├── models.py             # Swin Transformer + EfficientNet models
│   ├── train.py              # Training loop, fine-tuning strategy
│   ├── evaluate.py           # AUC, bootstrap testing, confusion matrix
│   ├── visualize.py          # Saliency maps (Grad-CAM / Grad-CAM++)
│   └── utils.py              # Helpers for metrics, LR scheduling, etc.
│
├── notebooks/
│   ├── Swin_Training.ipynb
│   ├── EfficientNet_Training.ipynb
│   └── Inference_and_Visualization.ipynb
│
├── requirements.txt
└── README.md

## 📦 Installation

Install dependencies:

```
pip install -r requirements.txt
```

Or for Colab:

```
!pip install timm einops albumentations pydicom scikit-learn
```

## 📊 Dataset: CheXpert (Small Subset)

We use **CheXpert-small**, which includes ~2% of the full CheXpert dataset.

**Target labels**
- Cardiomegaly
- Pneumonia
- Atelectasis
- Pleural Effusion
- No Finding

Only **frontal images** are used.  
Uncertain labels (`-1`) are mapped to **0**.

Patient-wise **80/20 split** prevents leakage.

## 🏗️ Model Architectures

### Swin Transformer (Swin-Tiny)
- Hierarchical windowed self-attention  
- ImageNet-pretrained  
- Last stage + classifier fine-tuned  

### EfficientNet-B0
- Lightweight, compound scaling  
- Fully fine-tuned  

## 🚀 Training Configuration

| Component      | Setting                    |
|----------------|----------------------------|
| Image Size     | 224×224                    |
| Optimizer      | AdamW                      |
| Scheduler      | Cosine Annealing           |
| Loss           | BCE + Focal Loss           |
| Batch Size     | 64                         |
| Epochs         | 10                         |
| Augmentation   | Flip, rotation, jitter, TA |
| Interpretability | Grad-CAM                 |

## 📈 Results Summary

Both models achieved similar performance due to:

- Limited number of epochs (10)
- Moderate dataset size (CheXpert-small)
- Low input resolution (224px)
- Global labels where CNNs perform strongly
