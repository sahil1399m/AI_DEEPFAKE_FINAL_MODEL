<h1 align="center">
  <a href="https://github.com/CommunityOfCoders/Inheritance2k25">
    CoC Inheritance 2025
  </a>
  <br>
  AIthentic: Military-Grade Neural Forensics
</h1>

<div align="center">
  By <strong>Team CodePagloos</strong>
</div>
<hr>

<details>
<summary><strong>Table of Contents</strong> (Click to Expand)</summary>

- [📝 Description](#-description)
- [🔗 Links](#-links)
- [🤖 Tech-Stack](#-tech-stack)
- [🏗️ System Architecture](#-system-architecture)
- [📈 Progress](#-progress)
- [🔮 Future Scope](#-future-scope)
- [💸 Applications](#-applications)
- [🛠 Project Setup](#-project-setup)
- [👨‍💻 Team Members](#-team-members)
- [👨‍🏫 Mentors](#-mentors)

</details>

<a name="-description"></a>
## 📝 Description

**AIthentic** is an enterprise-grade forensic platform designed to expose Deepfake media by analyzing temporal inconsistencies invisible to the human eye. Unlike traditional detectors that analyze static frames, our system uses **Active Entropy Sampling** to target high-motion segments and employs a **Hybrid Spatial-Temporal Network (EfficientNet-B3 + Bi-LSTM)** to detect micro-flickers and warping artifacts.

The platform achieves **96.71% accuracy** and features a cyber-forensic dashboard with a live neural terminal and an integrated AI assistant.

<a name="-links"></a>
## 🔗 Links

- [GitHub Repository](https://github.com/sahil1399m/AI_DEEPFAKE_FINAL_MODEL)
- [Demo Video]([INSERT_YOUR_YOUTUBE_LINK])
- [Project Screenshots]([INSERT_YOUR_DRIVE_LINK])
- [Hosted Website](https://aideepfakefinalmodel-8klv8esicpxfjwcq3b7jjj.streamlit.app/)

<a name="-tech-stack"></a>
## 🤖 Tech-Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, Torchvision
* **Model Architectures:**
    * `EfficientNet-B3` (Spatial Feature Extraction)
    * `Bi-Directional LSTM` (Temporal Sequence Modeling)
    * `MTCNN` (Face Detection)
* **Computer Vision:** OpenCV, PIL, Active Entropy Algorithms
* **Frontend/UI:** Streamlit, Custom CSS (Cyberpunk/SOC Theme), Plotly (Telemetry)
* **GenAI:** Google Gemini 1.5 Flash API (Forensic Assistant)
* **Data Handling:** NumPy, Pandas

<a name="-system-architecture"></a>
## 🏗️ System Architecture

Our architecture combines spatial feature extraction with temporal sequence analysis to detect anomalies that occur over time, not just in single frames.

```mermaid
graph TD
    subgraph "Input Layer"
      A[📹 INPUT VIDEO STREAM]:::input
    end

    subgraph "Preprocessing Engine"
      B(⚡ Active Entropy Sampling):::process
      C{👤 Face Detection & Alignment}:::process
    end

    subgraph "Deep Learning Core"
      D[👁️ EfficientNet-B3<br/><i>Spatial Analysis</i>]:::model
      E[🧠 Bi-Directional LSTM<br/><i>Temporal Analysis</i>]:::model
    end

    subgraph "Forensic Output"
      F[🛡️ CONFIDENCE SCORE]:::logic
      G[✅ REAL FOOTAGE]:::safe
      H[⚠️ DEEPFAKE DETECTED]:::alert
    end

    %% Flow Connections
    A -->|Extract Frames| B
    B -->|Top 20 High-Entropy Frames| C
    C -->|Cropped Faces 224x224| D
    D -->|Feature Vectors 1536-dim| E
    E -->|Sequence Analysis| F
    
    %% Logic Logic
    F -- Score < 0.5 --> G
    F -- Score > 0.5 --> H

    %% Styling Definitions (Cyberpunk Theme)
    classDef input fill:#000,stroke:#00ff41,stroke-width:2px,color:#00ff41;
    classDef process fill:#000,stroke:#2979ff,stroke-width:2px,color:#2979ff;
    classDef model fill:#1a1a2e,stroke:#d946ef,stroke-width:2px,color:#d946ef;
    classDef logic fill:#000,stroke:#fbbf24,stroke-width:2px,color:#fbbf24;
    classDef safe fill:#064e3b,stroke:#00ff41,stroke-width:2px,color:#fff;
    classDef alert fill:#450a0a,stroke:#ff1744,stroke-width:2px,color:#fff;
```

<a name="-progress"></a>
## 📈 Progress

### Fully Implemented Features
* **Active Entropy Sampling:** Successfully filters 90% of useless background frames to optimize inference speed (~2.5s per video).
* **Hybrid Detection Core:** The EfficientNet-B3 + Bi-LSTM architecture is fully trained and verified with **96.71% Test Accuracy**.
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
