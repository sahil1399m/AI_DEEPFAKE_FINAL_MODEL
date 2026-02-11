# 👁️ AIthentic: Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.32.0-red)
![Model](https://img.shields.io/badge/PyTorch-EfficientNet%2BBiLSTM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**AIthentic** is a military-grade forensic tool designed to detect deepfake videos using a hybrid **Spatial-Temporal** architecture. It combines **EfficientNet-B3** (for spatial feature extraction) and **Bi-Directional LSTMs** (for temporal consistency checking) to flag synthetic media with **96.71% accuracy**.

---

## 🚀 Features

* **🕵️‍♂️ Active Entropy Sampling:** Intelligent frame selection that ignores static backgrounds and focuses on high-motion facial areas.
* **🧠 Hybrid Neural Network:** * **Spatial:** EfficientNet-B3 (Pre-trained on ImageNet).
    * **Temporal:** Bi-Directional LSTM to detect "flicker" and "jitter" over time.
* **🖥️ Cyber-Forensic Dashboard:** A futuristic SOC-style UI built with Streamlit.
* **🤖 AI Assistant:** Integrated Google Gemini 1.5 Flash API for explaining forensic results (Optional).
* **📊 Live Telemetry:** Real-time system resource monitoring (simulated for UI).

---

## 🛠️ Installation Guide

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/sahil1399m/AI_DEEPFAKE_FINAL_MODEL.git](https://github.com/sahil1399m/AI_DEEPFAKE_FINAL_MODEL.git)
cd AI_DEEPFAKE_FINAL_MODEL* **Hybrid Detection Core:** The EfficientNet-B3 + Bi-LSTM architecture is fully trained and verified with **96.71% Test Accuracy**.
* **Forensic Dashboard:** A complete UI with "Bouncer" logic visualizations, file upload, and real-time "Live Terminal" logs.
* **Gemini Assistant:** Integrated chatbot that explains forensic verdicts (e.g., "Why is this fake?") to non-technical users.

<a name="-future-scope"></a>
## 🔮 Future Scope
* **Audio-Visual Sync:** Integration of Wav2Lip-based models to detect desynchronization between lip movements and audio.
* **Adversarial Defense:** Training against "anti-forensic" noise attacks (Gaussian blur/compression).
* **Browser Extension:** Lightweight plugin to flag synthetic media on X/Twitter in real-time.

<a name="-applications"></a>
## 💸 Applications
1. **Legal & Judiciary:** Authenticating video evidence in court to prevent tampering.
2. **News & Media:** Verifying political speeches and breaking news footage before broadcast.
3. **Digital Identity (KYC):** Detecting "virtual camera" injection attacks during remote video verification.

<a name="-project-setup"></a>
## 🛠 Project Setup

#### 1. Clone the Repository
```bash
git clone [https://github.com/sahil1399m/AI_DEEPFAKE_FINAL_MODEL](https://github.com/sahil1399m/AI_DEEPFAKE_FINAL_MODEL)
cd AIthentic
```
#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 3. Run the Application
```bash
streamlit run app.py
```

<a name="-team-members"></a>
## 👨‍💻 Team Members

* **Sahil Desai:** [GitHub Profile](https://github.com/SahilDesai-0)
* **Himanshu:** [GitHub Profile](https://github.com/SahilDesai-0)
* **Tejas:** [GitHub Profile](https://github.com/SahilDesai-0)
* **Krish:** [GitHub Profile](https://github.com/SahilDesai-0)

<a name="-mentors"></a>
## 👨‍🏫 Mentors

* **Abhishek Kotwani:** [GitHub/LinkedIn](https://github.com/SahilDesai-0)
* **Om Mukherjee:** [GitHub/LinkedIn](https://github.com/SahilDesai-0)
