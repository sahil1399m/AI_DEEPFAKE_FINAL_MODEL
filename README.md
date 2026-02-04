# 👁️ AIthentic: Military-Grade Neural Forensics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Cyberpunk%20UI-ff4b4b?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25-success?style=for-the-badge&logo=google-analytics)
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

## 🧠 Core Architecture (The "Secret Sauce")

Our system pipeline is divided into three military-grade processing stages.

### 1. ⚡ Active Entropy Sampling (AES)
* **The Logic:** 90% of a video frame is useless background. Processing it is a waste of GPU.
* **The Fix:** We use a pixel-difference algorithm to calculate the **Entropy** of every frame.
* **The Result:** The system ignores the noise and extracts only the **Top 20 High-Motion Frames** (e.g., mid-blink, lip purse, head turn)—the exact moments where Deepfakes glitch.

### 2. 👁️ Spatial Feature Extraction (EfficientNet-B3)
* We utilize a pre-trained **EfficientNet-B3** as our visual backbone.
* It converts raw pixels into dense **1536-dimensional feature vectors**.
* **Target:** It spots skin smoothing, resolution mismatches (FaceForensics++ c23 artifacts), and blending boundaries.

### 3. 🧠 Temporal Sequence Modeling (Bi-LSTM)
* The extracted vectors are fed into a **Bidirectional LSTM** (Long Short-Term Memory) network.
* **Why Bidirectional?** It analyzes the video *forwards* and *backwards* simultaneously to understand context.
* **Target:** **Temporal Jitter**—micro-flickering in the eyes or lips that happens when a Generator loses temporal coherence.

---

## 📊 Performance Benchmarks

We don't just guess; we prove it. The model was rigorously trained and evaluated on the **FaceForensics++** dataset (Deepfakes, Face2Face, FaceSwap).

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Test Accuracy** | **90.00%** | Evaluated on unseen `Real` vs `Fake` footage. |
| **Training Acc** | **98.5%** | After 25 epochs of extended fine-tuning. |
| **Inference Time** | **~2.5s** | Average processing time per video using Active Sampling. |

> *Data derived from internal testing on the FaceForensics++ (c23 compression) subset.*

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
