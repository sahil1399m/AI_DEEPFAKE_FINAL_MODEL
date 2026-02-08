# CoC Inheritance 2025
# System Health Metric Analyzer: Real-time Monitoring & Analysis
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
The System Health Metric Analyzer is a dual-platform application (Web & Desktop) designed to monitor, analyze, and visualize critical system performance metrics in real-time. It provides users with detailed insights into CPU usage, memory consumption, disk activity, and network traffic. By leveraging a robust backend with Supabase and secure Google OAuth authentication, the tool ensures data persistence and secure access across devices, empowering administrators and power users to maintain optimal system health.

## 🔗 Links
* [GitHub Repository]([INSERT_YOUR_GITHUB_REPO_LINK])
* [Demo Video]([INSERT_YOUR_YOUTUBE_OR_DRIVE_LINK])
* [Project Screenshots/Drive]([INSERT_YOUR_DRIVE_FOLDER_LINK])
* [Hosted Website]([INSERT_YOUR_STREAMLIT_SHARE_LINK])

## 🤖 Tech-Stack
* **Language:** Python 3.12.0
* **Frontend:** * **Web:** Streamlit
    * **Desktop:** PyQt5
* **Backend & Database:** Supabase (PostgreSQL)
* **Authentication:** Google OAuth 2.0
* **System Monitoring:** `psutil` library
* **Data Visualization:** Plotly, Matplotlib
* **Version Control:** Git & GitHub

## 🏗️ System Architecture

```mermaid
graph TD
    %% Nodes
    User[👤 User] -->|Authenticates| Auth[🔐 Google OAuth]
    Auth -->|Access Granted| App{🖥️ Application}
    
    subgraph "Application Layer"
        App -->|Web Access| Streamlit[🌐 Streamlit Web App]
        App -->|Desktop Access| PyQt[💻 PyQt5 Desktop App]
    end
    
    subgraph "Logic & Monitoring"
        Streamlit -->|Collects Metrics| PSUtil[⚙️ psutil Library]
        PyQt -->|Collects Metrics| PSUtil
    end
    
    subgraph "Data Layer"
        PSUtil -->|Writes Data| DB[(🗄️ Supabase Database)]
        DB -->|Reads Data| Streamlit
        DB -->|Reads Data| PyQt
    end

    %% Styling
    classDef user fill:#FFD700,stroke:#333,stroke-width:2px;
    classDef auth fill:#4285F4,stroke:#333,stroke-width:2px,color:#fff;
    classDef app fill:#34A853,stroke:#333,stroke-width:2px,color:#fff;
    classDef db fill:#EA4335,stroke:#333,stroke-width:2px,color:#fff;
    
    class User user;
    class Auth auth;
    class Streamlit,PyQt app;
    class DB db;
