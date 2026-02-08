# CoC Inheritance 2025
# AIthentic: Military-Grade Neural Forensics
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
AIthentic is an enterprise-grade forensic platform designed to expose Deepfake media by analyzing temporal inconsistencies invisible to the human eye. Unlike traditional detectors that analyze static frames, our system uses **Active Entropy Sampling** to target high-motion segments and employs a **Hybrid Spatial-Temporal Network (EfficientNet-B3 + Bi-LSTM)** to detect micro-flickers and warping artifacts. The platform achieves **96.71% accuracy** and features a cyber-forensic dashboard with a live neural terminal and an integrated AI assistant.

## 🔗 Links
* [GitHub Repository]([INSERT_YOUR_GITHUB_REPO_LINK])
* [Demo Video]([INSERT_YOUR_YOUTUBE_OR_DRIVE_LINK])
* [Project Screenshots/Drive]([INSERT_YOUR_DRIVE_FOLDER_LINK])
* [Hosted Website]([INSERT_YOUR_STREAMLIT_SHARE_LINK])

## 🤖 Tech-Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, Torchvision
* **Model Architectures:** EfficientNet-B3 (Spatial Feature Extraction), Bi-Directional LSTM (Temporal Sequence Modeling), MTCNN (Face Detection)
* **Computer Vision:** OpenCV, PIL, Active Entropy Algorithms
* **Frontend/UI:** Streamlit, Custom CSS (Cyberpunk/SOC Theme), Plotly (Telemetry)
* **GenAI:** Google Gemini 1.5 Flash API (Forensic Assistant)
* **Data Handling:** NumPy, Pandas

## 🏗️ System Architecture

### 📡 Data Pipeline Flow
```text
[ 📹 INPUT STREAM ] 
       │
       ▼
( ⚡ Active Entropy Sampling ) ──▶ Filters 90% Static Frames
       │
       ▼
{ 👤 MTCNN Face Detection } ───▶ Crops & Aligns Faces
       │
       ▼
[ 👁️ EfficientNet-B3 ] ──────▶ Extracts Spatial Features (1536-dim vectors)
       │
       ▼
[ 🧠 Bi-Directional LSTM ] ───▶ Analyzes Temporal Jitter (Forward/Backward)
       │
       ▼
[ 🛡️ CONFIDENCE SCORE ] ──────▶ Verdict: REAL vs FAKE
