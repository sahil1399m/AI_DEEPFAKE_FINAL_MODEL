# CoC Inheritance 2025
# AIthentic: Military-Grade Neural Forensics
By ** Team CodePagloos**

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
AIthentic is an enterprise-grade forensic platform designed to expose Deepfake media by analyzing temporal inconsistencies invisible to the human eye. Unlike traditional detectors that analyze static frames, our system uses **Active Entropy Sampling** to target high-motion segments and employs a **Hybrid Spatial-Temporal Network (EfficientNet-B3 + Bi-LSTM)** to detect micro-flickers and warping artifacts. We achieve **96.71% accuracy** on the FaceForensics++ benchmark by focusing on the "glitch in time" rather than just static pixels.

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

### Front-end
* **Streamlit Interface:** A "Cyber-Forensic" dashboard designed for Security Operations Centers (SOCs).
* **Live Neural Terminal:** A simulated CLI that provides real-time transparency into the model's inference steps (tensor allocation, weight loading).
* **Visuals:** Integrated Lottie animations and Neon Telemetry graphs for confidence scoring.

### Back-end
* **Inference Engine:** PyTorch pipeline utilizing `efficientnet_b3_lstm_active.pth` weights.
* **Active Entropy Sampling (AES):** A logic layer that calculates pixel-difference entropy to ignore 90% of static frames and target the top 20 high-motion moments (blinking, talking).

### Machine Learning Pipeline
1.  **Input Processing:** Video stream is scanned for high-entropy motion.
2.  **Spatial Analysis (The Eye):** EfficientNet-B3 strips classification heads to extract 1536-dimensional feature vectors, spotting skin warping and blending artifacts.
3.  **Temporal Analysis (The Memory):** Bi-Directional LSTM analyzes vector sequences forward and backward to detect "Temporal Jitter" (micro-flickering).

## 📈 Progress

### Fully Implemented Features
* **Active Entropy Sampling:** Successfully filters useless background frames to optimize inference speed (~2.5s per video).
* **Hybrid Detection Core:** The EfficientNet-B3 + Bi-LSTM architecture is fully trained and verified with **96.71% Test Accuracy** and **0.99 Precision**.
* **Forensic Dashboard:** A complete UI with "Bouncer" logic visualizations, file upload, and real-time terminal logs.
* **Gemini Assistant:** Integrated chatbot that explains forensic verdicts (e.g., "Why is this fake?") to non-technical users.

### Partially Implemented Features / Work in Progress
* **Advanced Telemetry:** Detailed breakdown of specific artifact types (Lighting vs. Texture) is currently approximated; moving towards granular head-specific detection.
* **Edge Deployment:** Work is underway to quantize the model for mobile/CCTV deployment without losing accuracy.

## 🔮 Future Scope
* **Audio-Visual Sync:** Integration of Wav2Lip-based models to detect desynchronization between lip movements and audio tracks.
* **Adversarial Defense:** Training the model against "anti-forensic" noise attacks to ensure robustness in the wild.
* **Browser Extension:** A lightweight plugin to flag synthetic media on social platforms (X/Twitter, LinkedIn) in real-time.

## 💸 Applications
* **Legal & Judiciary:** Authenticating video evidence in court to prevent tampering or fabrication.
* **News & Media:** verifying political speeches and breaking news footage before broadcast to combat disinformation.
* **Digital Identity (KYC):** Enhancing banking security by detecting "virtual camera" injection attacks during remote video verification.

## 🛠 Project Setup

Clone the GitHub repo.
```bash
git clone [INSERT_YOUR_GITHUB_REPO_LINK]
