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
cd AI_DEEPFAKE_FINAL_MODEL
```
### 2. Create a Virtual Environment 
```bash
# Windows 
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
## 🤖 Tech Stack

* **Frontend:** Streamlit, Custom CSS (Cyberpunk Theme)
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV (cv2), MTCNN (Face Detection)
* **Data Processing:** NumPy, Pandas, PIL
* **GenAI:** Google Generative AI (Gemini)

---

## 🤝 Contributing

We welcome contributions to improve the forensic accuracy of AIthentic!

1.  **Fork the repository** to your GitHub account.
2.  **Create a new branch** for your feature:
    ```bash
    git checkout -b feature-branch
    ```
3.  **Commit your changes** with descriptive messages:
    ```bash
    git commit -m "Added new entropy algorithm"
    ```
4.  **Push to the branch**:
    ```bash
    git push origin feature-branch
    ```
5.  **Open a Pull Request** and describe your changes.

---

<div align="center">
  Made with ❤️ by <strong>Team CodePagloos</strong>
</div>
