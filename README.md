# 👁️ AIthentic: Military-Grade Neural Forensics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Cyberpunk%20UI-ff4b4b?style=for-the-badge&logo=streamlit)
![Google Gemini](https://img.shields.io/badge/AI-Gemini%20Assistant-8E75B2?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **"Truth in the age of Synthetic Media."**

**AIthentic** is an enterprise-grade forensic platform designed to expose Deepfakes. Unlike standard detectors that look at static images, AIthentic analyzes the **temporal heartbeat** of a video—catching the micro-flickers, jitters, and texture anomalies that generative AI cannot yet hide.

---

## 🧠 How It Works (The "Secret Sauce")

Most detectors fail because they analyze random frames. We don't. We use **Active Entropy Sampling**.

We scan the video for high-motion moments (blinking, talking, head turning). If a Deepfake is going to glitch, it will glitch there. We extract those specific frames and feed them into a hybrid brain:

1.  **The Eye (Spatial):** `EfficientNet-B3` looks at individual pixels for skin warping and blending artifacts.
2.  **The Memory (Temporal):** `Bi-Directional LSTM` looks at time. It watches the video forward and backward to see if the face moves naturally or "jitters."

### 📡 System Pipeline

```mermaid
graph TD
    A[📹 Input Video] -->|Pixel Difference Algo| B[⚡ Active Entropy Sampling]
    B -->|Extract Top 20 High-Motion Frames| C[🖼️ ROI Face Extraction]
    C -->|Texture Analysis| D[👁️ EfficientNet-B3 CNN]
    D -->|Feature Vector Sequence| E[🧠 Bi-Directional LSTM]
    E -->|Temporal Analysis| F{🛡️ FINAL VERDICT}
    F -->|Real| G[✅ AUTHENTIC]
    F -->|Fake| H[⚠️ DEEPFAKE DETECTED]
