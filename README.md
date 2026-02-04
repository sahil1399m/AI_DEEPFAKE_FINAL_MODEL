# 👁️ AIthentic: Military-Grade Neural Forensics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Cyberpunk%20UI-ff4b4b?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.71%25-success?style=for-the-badge&logo=google-analytics)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <strong>Detecting the invisible heartbeat of Synthetic Media.</strong>
</p>

</div>

---

## 🚨 The Reality
Deepfakes are no longer just "blurry faces." They are generating perfect pixels. Traditional detectors that look at a single frame fail because **modern GANs don't make spatial mistakes—they make temporal ones.**

## 🛡️ The AIthentic Solution
**AIthentic** ignores the static image. Instead, it watches the **temporal vector field**—the tiny, invisible "jitters" that occur when a neural network tries to generate movement over time. We built a custom **Spatial-Temporal Engine** that surgically targets high-entropy motion (blinking, talking) to expose the fake.

---

## 🔬 Scientific Foundation & Architecture

Our architecture is not random; it is a hybrid implementation derived from state-of-the-art forensic research.

### 1. ⚡ Active Entropy Sampling (AES)
* **The Logic:** 90% of a video frame is useless background. Processing it is a waste of GPU.
* **The Fix:** We use a pixel-difference algorithm to calculate the **Entropy** of every frame.
* **The Result:** The system ignores static noise and surgically extracts the **Top 20 High-Motion Frames** (e.g., mid-blink, lip purse, head turn)—the exact moments where Deepfakes glitch.

### 2. 👁️ Spatial Feature Extraction (EfficientNet-B3)
* **Research Reference:** Inspired by *Li et al. (CVPRW 2019)*, "Exposing DeepFake Videos By Detecting Face Warping Artifacts".
* **Theory:** Deepfakes are limited by resolution and affine warping. Even if the face looks real, the *warping* needed to fit it onto the target head leaves specific artifacts at the boundaries.
* **Implementation:** We use a pre-trained **EfficientNet-B3** to extract a 1536-dimensional feature vector that encodes these subtle texture and warping anomalies.

### 3. 🧠 Temporal Sequence Modeling (Bi-LSTM)
* **Research Reference:** Inspired by *Liu et al. (WACV 2023)*, "TI2Net: Temporal Identity Inconsistency Network".
* **Theory:** Fake videos suffer from "Temporal Identity Inconsistency"—the face in Frame 1 is mathematically slightly different from the face in Frame 2.
* **Implementation:** We feed the spatial vectors into a **Bidirectional LSTM**. It analyzes the video *forwards and backwards*, catching the **Temporal Jitter** (micro-flickering) that occurs when a generator loses coherence over time.

---

## 📊 Performance Benchmarks (Verified)

Our model was rigorously trained and evaluated on the **FaceForensics++ (c23)** and **Celeb-DF-v2** datasets.

| Metric | Result | Source |
| :--- | :--- | :--- |
| **Test Accuracy** | **96.71%** | `FINAL-MODEL-TEST.pdf` |
| **Training Acc** | **99.42%** | `FINAL-MODEL-TRAIN.pdf` |
| **Validation Acc** | **96.67%** | `FINAL-MODEL-TRAIN.pdf` |
| **Precision (Fake)**| **0.99** | Minimal False Positives |
| **Recall (Real)** | **0.99** | Minimal False Negatives |

> *Evaluated on 152 unseen test videos using a sequence length of 20 frames.*

---

## 📡 Forensic Pipeline Visualization

```mermaid
graph TD
    A[📹 Input Stream] -->|Entropy Algorithm| B(⚡ Active Sampling)
    B -->|Filter Best 20 Frames| C{MTCNN Face Detect}
    C -->|Aligned Faces| D[👁️ EfficientNet-B3]
    D -->|1536-dim Vectors| E[🧠 Bi-Directional LSTM]
    E -->|Temporal Analysis| F[🛡️ CONFIDENCE SCORE]
    
    F -->|Score < 0.5| G[✅ REAL FOOTAGE]
    F -->|Score > 0.5| H[⚠️ DEEPFAKE DETECTED]
    
    style A fill:#00ff41,stroke:#333,stroke-width:2px,color:#000
    style H fill:#ff003c,stroke:#333,stroke-width:2px,color:#fff
    style G fill:#00ff41,stroke:#333,stroke-width:2px,color:#000
