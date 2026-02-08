# CoC Inheritance 2025
# 🛡️ AIthentic: Military-Grade Neural Forensics
By **Team CodePagloos**

## Table of Contents
* [📝 Description](#-description)
* [🔗 Links](#-links)
* [🤖 Tech-Stack](#-tech-stack)
* [🏗️ System Architecture](#-system-architecture)
* [📈 Progress](#-progress)
* [🔮 Future Scope](#-future-scope)
* [💸 Applications](#-applications)
* [🛠 Project Setup](#-project-setup)
* [👨‍💻 Team Members](#-team-members)
* [👨‍🏫 Mentors](#-mentors)

## 📝 Description
**AIthentic** is an enterprise-grade forensic platform designed to expose Deepfake media by analyzing temporal inconsistencies invisible to the human eye. Unlike traditional detectors that analyze static frames, our system uses **Active Entropy Sampling** to target high-motion segments and employs a **Hybrid Spatial-Temporal Network (EfficientNet-B3 + Bi-LSTM)** to detect micro-flickers and warping artifacts. The platform achieves **96.71% accuracy** and features a cyber-forensic dashboard with a live neural terminal and an integrated AI assistant.

## 🔗 Links
* [GitHub Repository]([INSERT_YOUR_GITHUB_REPO_LINK])
* [Demo Video]([INSERT_YOUR_YOUTUBE_OR_DRIVE_LINK])
* [Project Screenshots/Drive]([INSERT_YOUR_DRIVE_FOLDER_LINK])
* [Hosted Website]([INSERT_YOUR_STREAMLIT_SHARE_LINK])

## 🤖 Tech-Stack
* **Language:** `Python 3.10+`
* **Deep Learning:** `PyTorch`, `Torchvision`
* **Model Architectures:** `EfficientNet-B3` (Spatial), `Bi-Directional LSTM` (Temporal), `MTCNN` (Face Detect)
* **Computer Vision:** `OpenCV`, `PIL`, `Active Entropy Algos`
* **Frontend/UI:** `Streamlit`, `Custom CSS` (Cyberpunk/SOC Theme), `Plotly` (Telemetry)
* **GenAI:** `Google Gemini 1.5 Flash API` (Forensic Assistant)
* **Data Handling:** `NumPy`, `Pandas`

## 🏗️ System Architecture

### 📡 Data Pipeline Visualization
```text
╔═════════════════════════════════════════════════════════════════════╗
║                       📹  INPUT VIDEO STREAM                        ║
╚═════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ⚡ ACTIVE ENTROPY SAMPLING (Preprocessing Layer)                   │
│  • Calculates Pixel-Difference Entropy                              │
│  • Discards 90% Static Frames -> Keeps Top 20 High-Motion Frames    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  👤 FACE EXTRACTION ENGINE                                          │
│  • MTCNN (Multi-Task Cascaded CNN)                                  │
│  • Cropping, Alignment & Normalization (224x224px)                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐
│  👁️ SPATIAL ANALYSIS         │       │  🧠 TEMPORAL ANALYSIS        │
│  (EfficientNet-B3)           │       │  (Bi-Directional LSTM)       │
│                              │       │                              │
│  Extracts 1536-dim vectors   │ ────▶ │  Analyzes Sequence Fwd/Bwd   │
│  Detects: Blending/Warping   │       │  Detects: Temporal Jitter    │
└──────────────────────────────┘       └──────────────────────────────┘
                                                  │
                                                  ▼
                                   ╔══════════════════════════════════╗
                                   ║    🛡️ FINAL VERDICT: SCORE       ║
                                   ║    (Real < 0.5 < Fake)           ║
                                   ╚══════════════════════════════════╝

### Machine Learning Pipeline
1.  **Input Processing:** Video stream is scanned for high-entropy motion using pixel-difference algorithms.
2.  **Spatial Analysis (The Eye):** EfficientNet-B3 strips classification heads to extract 1536-dimensional feature vectors, spotting skin warping and blending artifacts.
3.  **Temporal Analysis (The Memory):** Bi-Directional LSTM analyzes vector sequences forward and backward to detect "Temporal Jitter" (micro-flickering).

## 📈 Progress

### Fully Implemented Features
* **Active Entropy Sampling:** Successfully filters 90% of useless background frames to optimize inference speed (~2.5s per video).
* **Hybrid Detection Core:** The EfficientNet-B3 + Bi-LSTM architecture is fully trained and verified with **96.71% Test Accuracy** and **0.99 Precision**.
* **Forensic Dashboard:** A complete UI with "Bouncer" logic visualizations, file upload, and real-time "Live Terminal" logs.
* **Gemini Assistant:** Integrated chatbot that explains forensic verdicts (e.g., "Why is this fake?") to non-technical users.

### Partially Implemented Features / Work in Progress
* **Advanced Telemetry:** Detailed breakdown of specific artifact types (Lighting vs. Texture) is currently approximated; moving towards granular head-specific detection.
* **Edge Deployment:** Work is underway to quantize the model for mobile/CCTV deployment to reduce size without losing accuracy.

## 🔮 Future Scope
* **Audio-Visual Sync:** Integration of Wav2Lip-based models to detect desynchronization between lip movements and audio tracks.
* **Adversarial Defense:** Training the model against "anti-forensic" noise attacks (Gaussian blur/compression) to ensure robustness in the wild.
* **Browser Extension:** Developing a lightweight browser plugin to flag synthetic media on social platforms (X/Twitter, LinkedIn) in real-time.

## 💸 Applications
* **Legal & Judiciary:** Authenticating video evidence in court to prevent tampering or fabrication.
* **News & Media:** Verifying political speeches and breaking news footage before broadcast to combat disinformation.
* **Digital Identity (KYC):** Enhancing banking security by detecting "virtual camera" injection attacks during remote video verification.

## 🛠 Project Setup

**1. Clone the GitHub repo**
```bash
git clone [INSERT_YOUR_GITHUB_REPO_LINK]
