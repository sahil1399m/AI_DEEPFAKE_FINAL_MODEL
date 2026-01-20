# 🛡️ AIthentic: Enterprise Deepfake Forensic Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-ff4b4b?style=for-the-badge&logo=streamlit)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI%20Assistant-8E75B2?style=for-the-badge&logo=google)

**AIthentic** is a state-of-the-art Neural Forensic Platform designed to detect deepfake media manipulation using **Active Temporal Sampling** and **Recurrent Neural Networks**. Unlike traditional frame-by-frame classifiers, AIthentic focuses on high-entropy motion frames where deepfake artifacts (blending, jitter) are most likely to occur.

---

## 🚀 Key Features

### 1. Active Temporal Sampling
Instead of analyzing every frame (which is slow and redundant), AIthentic scans the video's **Temporal Vector Field** to identify the top 20 frames with the highest motion entropy. This targets the analysis specifically on moments where the subject is talking, blinking, or turning—the exact moments where Deepfake models fail.

### 2. Hybrid Neural Architecture
* **Spatial Analysis (EfficientNet-B3):** Extracts high-fidelity texture features from individual frames to spot skin-blending artifacts.
* **Temporal Analysis (Bi-Directional LSTM):** Tracks feature consistency across time to detect "temporal jitter" (flickering) that is invisible to the naked eye.

### 3. AI Forensic Assistant
Integrated with **Google Gemini 1.5 Flash**, the platform includes a context-aware chatbot that explains forensic concepts, details the detection methodology, and guides users through the analysis process in real-time.

---

## 🛠️ System Architecture

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Responsive, neon-styled forensic dashboard. |
| **Face Detection** | MTCNN | Extracting faces from high-motion frames. |
| **Feature Extraction** | EfficientNet-B3 | analyzing texture and compression artifacts. |
| **Sequence Modeling** | Bi-LSTM | Detecting temporal inconsistencies (glitches over time). |
| **GenAI Integration** | Google Gemini API | Interactive forensic assistant. |

---

## 📂 Project Structure

```text
AI-DEEPFAKE-FINAL-MODEL/
├── assets/                  # UI assets (Lottie animations, backgrounds)
├── Codebooks/               # Research & Training Notebooks (.ipynb)
├── model_weights/           # Trained PyTorch models (ignored by git)
├── .gitignore               # Security & cleanup settings
├── app.py                   # Main Application Entry Point
├── requirements.txt         # Project Dependencies
└── README.md                # Documentation
