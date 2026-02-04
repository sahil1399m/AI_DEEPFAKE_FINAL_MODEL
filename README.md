# 👁️ AIthentic: Military-Grade Neural Forensics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Cyberpunk%20UI-ff4b4b?style=for-the-badge&logo=streamlit)
![Google Gemini](https://img.shields.io/badge/AI-Gemini%20Assistant-8E75B2?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <strong>Detecting the invisible heartbeat of Synthetic Media.</strong>
</p>

</div>

---

## 🚨 The Problem
Deepfakes are getting scary good. Traditional detectors look at a single frame and ask: *"Does this face look fake?"*
**This fails** because modern GANs generate perfect static faces. The flaw isn't in the *pixels*; it's in the *timing*.

## 🛡️ The AIthentic Solution
**AIthentic** does not just look at the face; it watches how the face *moves* over time. We built a custom **Spatial-Temporal Engine** that targets high-entropy motion (blinking, talking) to catch the micro-jitters that Generative AI cannot hide yet.

---

## 🧠 Core Architecture (The "Secret Sauce")

Our system pipeline is divided into three military-grade processing stages.

### 1. ⚡ Active Entropy Sampling (AES)
* **The Issue:** Analyzing every frame of a video is slow and useless (90% of frames are static).
* **Our Fix:** We calculate the **Pixel-Difference Entropy** for the entire video stream.
* **The Result:** The system ignores static noise and surgically extracts the **Top 20 High-Motion Frames**—the exact moments (head turns, lip movements) where Deepfakes glitch.

### 2. 👁️ Spatial Feature Extraction (EfficientNet-B3)
* We strip the classification head off a pre-trained **EfficientNet-B3**.
* It acts as a feature extractor, converting raw pixels into dense **1536-dimensional vectors**.
* It detects **Spatial Artifacts:** Blending boundaries, skin warping, and resolution mismatches.

### 3. 🧠 Temporal Sequence Modeling (Bi-LSTM)
* The sequence of vectors is fed into a **Bidirectional LSTM** (Long Short-Term Memory) network.
* **Why Bidirectional?** It analyzes the video *forwards* and *backwards* simultaneously.
* It detects **Temporal Jitter:** Micro-flickering in the eyes or lips that occurs when a generator loses temporal coherence.

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
    
    style A fill:#0f0,stroke:#333,stroke-width:2px
    style H fill:#f00,stroke:#333,stroke-width:2px
