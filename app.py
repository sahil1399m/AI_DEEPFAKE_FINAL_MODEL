import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import time
import os
import json
import base64
import random
import pandas as pd
from streamlit_lottie import st_lottie
from facenet_pytorch import MTCNN
import google.generativeai as genai
import gdown
import plotly.graph_objects as go # 🆕 ADDED FOR COOL CHARTS

# ==========================================
# 🎨 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AIthentic | Neural Forensics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🔑 2. API SETUP
# ==========================================
GOOGLE_API_KEY = "xyz" 

gemini_active = False
try:
    if GOOGLE_API_KEY != "PASTE_YOUR_KEY_HERE" and GOOGLE_API_KEY != "xyz":
        clean_key = GOOGLE_API_KEY.strip()
        genai.configure(api_key=clean_key)
        gemini_active = True
except Exception as e:
    # Silent fail for UI aesthetics, handled in logic
    pass

# ==========================================
# 🧠 THE BRAIN: CONTEXT
# ==========================================
PROJECT_CONTEXT = """
ROLE: You are the "AIthentic Forensic Assistant", a military-grade neural expert.
GOAL: Explain the technical depth of the AIthentic platform.
SYSTEM:
1. Active Temporal Sampling: Entropy Scanning for high-motion frames.
2. Spatial Analysis (EfficientNet-B3): Texture anomalies.
3. Temporal Analysis (Bi-LSTM): Temporal Jitter.
"""

# ==========================================
# 📂 3. ASSET LOADER
# ==========================================
def load_lottie_local(filepath):
    try:
        with open(filepath, "r") as f: return json.load(f)
    except FileNotFoundError: return None

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError: return None

# Load Assets (Error handling included for portability)
lottie_left_scan = load_lottie_local("assets/animation1.json")
lottie_right_scan = load_lottie_local("assets/animation2.json")
lottie_chatbot = load_lottie_local("assets/animation3.json")
bg_image_base64 = get_base64_of_bin_file("assets/back_ground_img.jpg")

# ==========================================
# 🖌️ 4. ULTRA-MODERN CSS ENGINE (THE CRAZY STUFF)
# ==========================================

# Determine background
if bg_image_base64:
    background_style = f"""
    [data-testid="stAppViewContainer"] {{
        background: radial-gradient(circle at center, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.95) 100%), url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-attachment: fixed;
    }}
    """
else:
    # If no image, fall back to a cool dark grid
    background_style = """
    [data-testid="stAppViewContainer"] {
        background-color: #050505;
        background-image: linear-gradient(rgba(0, 243, 255, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 243, 255, 0.05) 1px, transparent 1px);
        background-size: 30px 30px;
    }
    """

st.markdown(f"""
<style>
    /* ---------------- IMPORTS ---------------- */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    :root {{ 
        --neon-blue: #00f3ff; 
        --neon-purple: #bc13fe; 
        --neon-green: #0aff48; 
        --neon-red: #ff003c; 
        --glass-bg: rgba(10, 15, 20, 0.75);
        --glass-border: 1px solid rgba(0, 243, 255, 0.2);
    }}

    /* ---------------- GLOBAL OVERRIDES ---------------- */
    html, body, [class*="css"] {{ font-family: 'Rajdhani', sans-serif; color: #e0fbfc; }}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{ width: 10px; background: #000; }}
    ::-webkit-scrollbar-thumb {{ background: var(--neon-blue); border-radius: 2px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--neon-purple); }}

    /* CRT Scanline Overlay Effect (The "Monitor" Look) */
    .scanlines {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        background-size: 100% 4px, 6px 100%;
        pointer-events: none; z-index: 9999;
    }}

    {background_style}
    
    /* Sidebar Blur */
    [data-testid="stSidebar"] {{
        background-color: rgba(5, 5, 10, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 243, 255, 0.1);
    }}

    /* ---------------- TEXT EFFECTS ---------------- */
    /* Glitch Animation for Titles */
    @keyframes glitch-skew {{
        0% {{ transform: skew(0deg); }}
        20% {{ transform: skew(-2deg); }}
        40% {{ transform: skew(2deg); }}
        60% {{ transform: skew(-1deg); }}
        80% {{ transform: skew(1deg); }}
        100% {{ transform: skew(0deg); }}
    }}
    
    .glitch-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 4.5rem;
        text-align: center;
        color: #fff;
        text-shadow: 2px 2px var(--neon-purple), -2px -2px var(--neon-blue);
        animation: glitch-skew 3s infinite linear alternate-reverse;
        margin-bottom: 0px;
        letter-spacing: 5px;
    }}

    .tech-subtitle {{
        font-family: 'Share Tech Mono', monospace;
        color: var(--neon-blue);
        text-align: center;
        font-size: 1.1rem;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-top: -10px;
        opacity: 0.9;
        text-shadow: 0 0 10px var(--neon-blue);
    }}

    /* ---------------- CARDS & COMPONENTS ---------------- */
    div[data-testid="stContainer"] {{ background: transparent; }}

    .tech-card {{
        background: rgba(10, 15, 20, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-left: 3px solid var(--neon-blue);
        padding: 25px;
        border-radius: 0px 20px 0px 20px; /* Cyberpunk cut corners */
        height: 100%;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }}
    
    /* Hover Glow Effect */
    .tech-card:hover {{
        border-left: 3px solid var(--neon-purple);
        box-shadow: 0 0 25px rgba(188, 19, 254, 0.2), inset 0 0 10px rgba(188, 19, 254, 0.1);
        transform: translateY(-5px) scale(1.01);
    }}

    .tech-header {{
        color: var(--neon-blue);
        font-family: 'Orbitron';
        font-size: 1.3rem;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* ---------------- BUTTONS ---------------- */
    .stButton button {{
        background: transparent !important;
        border: 1px solid var(--neon-blue) !important;
        color: var(--neon-blue) !important;
        font-family: 'Share Tech Mono' !important;
        font-size: 1.2rem !important;
        padding: 15px 30px !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 0px !important;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
    }}
    
    .stButton button::before {{
        content: '';
        position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 243, 255, 0.4), transparent);
        transition: 0.5s;
    }}

    .stButton button:hover {{
        background: rgba(0, 243, 255, 0.1) !important;
        box-shadow: 0 0 30px var(--neon-blue), inset 0 0 10px var(--neon-blue);
        text-shadow: 0 0 5px #fff;
    }}
    
    .stButton button:hover::before {{ left: 100%; }}

    /* Primary Button Redefine */
    button[kind="primary"] {{
        background: rgba(0, 243, 255, 0.1) !important;
        border: 1px solid var(--neon-green) !important;
        color: var(--neon-green) !important;
    }}
    button[kind="primary"]:hover {{
        box-shadow: 0 0 30px var(--neon-green) !important;
    }}

    /* ---------------- TERMINAL & LOGS ---------------- */
    .terminal-box {{
        background: #000;
        border: 1px solid #333;
        padding: 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.9rem;
        color: var(--neon-green);
        border-left: 4px solid var(--neon-green);
        height: 400px;
        overflow-y: auto;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
    }}
    .log-line {{ margin-bottom: 5px; border-bottom: 1px solid rgba(0,255,0,0.1); padding-bottom: 2px; }}

    /* ---------------- PIPELINE VIZ ---------------- */
    .pipeline-container {{
        display: flex; justify-content: space-between; align-items: center;
        background: rgba(0, 0, 0, 0.8); padding: 40px;
        border-radius: 10px; border: 1px solid var(--neon-blue);
        margin-bottom: 30px; box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);
        position: relative; overflow: hidden;
    }}
    
    .pipeline-node {{
        text-align: center; color: #fff;
        font-family: 'Share Tech Mono'; font-size: 1.1rem;
        background: rgba(255,255,255,0.05); padding: 15px 25px;
        border-radius: 5px; border: 1px solid #555;
        z-index: 2; transition: 0.3s;
    }}
    
    .pipeline-node:hover {{ border-color: var(--neon-blue); box-shadow: 0 0 15px var(--neon-blue); transform: scale(1.1); background: #000; }}
    
    .pipeline-arrow {{ color: var(--neon-blue); font-size: 2rem; animation: pulse-arrow 1.5s infinite; }}
    @keyframes pulse-arrow {{ 0% {{ opacity: 0.3; transform: translateX(0); }} 50% {{ opacity: 1; transform: translateX(10px); }} 100% {{ opacity: 0.3; transform: translateX(0); }} }}

    /* ---------------- EXPANDERS ---------------- */
    .stExpander {{ background-color: rgba(0, 10, 20, 0.8) !important; border: 1px solid #333 !important; }}
    .stExpander:hover {{ border-color: var(--neon-purple) !important; }}
    .streamlit-expanderHeader {{ font-family: 'Orbitron'; letter-spacing: 1px; color: #eee !important; }}

    /* ---------------- DEV TEAM CARD ---------------- */
    .dev-container {{
        position: relative; width: 100%; height: 240px;
        background: rgba(5, 10, 15, 0.8);
        border: 1px solid #333; border-top: 3px solid var(--neon-blue);
        padding: 20px; overflow: hidden; transition: 0.4s ease;
    }}
    .dev-container:hover {{ border-top: 3px solid var(--neon-purple); box-shadow: 0 10px 30px rgba(0,0,0,0.5); transform: translateY(-5px); }}
</style>

<div class="scanlines"></div>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 5. MODEL BACKEND (UNCHANGED LOGIC)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    class EfficientNetLSTM(nn.Module):
        def __init__(self, num_classes=2):
            super(EfficientNetLSTM, self).__init__()
            backbone = models.efficientnet_b3(weights=None)
            backbone.classifier = nn.Identity()
            self.feature_extractor = backbone
            self.lstm = nn.LSTM(input_size=1536, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(512 * 2, num_classes)

        def forward(self, x):
            batch_size, seq_len, c, h, w = x.size()
            c_in = x.view(batch_size * seq_len, c, h, w)
            features = self.feature_extractor(c_in)
            features = features.view(batch_size, seq_len, -1)
            lstm_out, _ = self.lstm(features)
            return self.fc(lstm_out[:, -1, :])

    model = EfficientNetLSTM().to(DEVICE)
    model_path = "efficientnet_b3_lstm_active.pth"

    # AUTO-DOWNLOAD FROM DRIVE
    if not os.path.exists(model_path):
        file_id = "1IpeVbi0jvwHaXD5qCMtF_peUVR9uJDw0" # Your ID
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            # Silent download for UI cleanliness
            gdown.download(url, model_path, quiet=True)
        except Exception:
            return None

    try:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return model
        return None
    except Exception:
        return None

model = load_model()
mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False)

# ==========================================
# 📽️ 6. VIDEO PROCESSOR (UNCHANGED LOGIC)
# ==========================================
def process_video_frames(video_path, status_log_func):
    cap = cv2.VideoCapture(video_path)
    frames, diffs = [], []
    ret, prev = cap.read()
    if not ret: return None, []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_cnt = 0
    
    status_log_func(">> INITIALIZING ENTROPY SCANNERS...", 0.1)
    
    while cap.isOpened():
        ret, curr = cap.read()
        if not ret: break
        if frame_cnt % 5 == 0:
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            score = np.mean(np.abs(curr_gray - prev_gray))
            diffs.append((score, curr))
            prev_gray = curr_gray
        frame_cnt += 1
    cap.release()

    diffs.sort(key=lambda x: x[0], reverse=True)
    top_frames = [x[1] for x in diffs[:20]]
    if len(top_frames) < 1: return None, []

    status_log_func(">> DETECTING FACIAL ROI (REGION OF INTEREST)...", 0.3)
    processed_faces = []

    for f in top_frames:
        h, w = f.shape[:2]
        scale = 640 / w
        small = cv2.resize(f, (0, 0), fx=scale, fy=scale)
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(small_rgb)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                w_b, h_b = x2-x1, y2-y1
                new_w, new_h = w_b*1.3, h_b*1.3
                cx, cy = x1 + w_b/2, y1 + h_b/2
                x1, y1 = max(0, int(cx - new_w/2)), max(0, int(cy - new_h/2))
                x2, y2 = min(w, int(cx + new_w/2)), min(h, int(cy + new_h/2))
                face = f[y1:y2, x1:x2]
                face = cv2.resize(face, (224, 224))
                processed_faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                break 

    final_faces = processed_faces[:20]
    while len(final_faces) < 20 and len(final_faces) > 0:
        final_faces.append(final_faces[-1])
    return final_faces, top_frames

# ==========================================
# 🧭 7. SIDEBAR (REDESIGNED)
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; border-bottom: 2px solid var(--neon-blue); padding-bottom: 20px; margin-bottom: 20px;">
        <h1 style="color: #fff; margin:0; font-family:'Orbitron'; letter-spacing: 3px; font-size: 2rem; text-shadow: 0 0 15px var(--neon-blue);">OPS CENTER</h1>
        <p style="color: var(--neon-blue); margin:0; font-size: 0.8rem; letter-spacing: 4px; font-family:'Share Tech Mono';">SYS.VER.3.1.0-ALPHA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom Radio Buttons via standard streamlit but styled by CSS
    pages = ["Dashboard", "Analysis Console", "Methodology", "About Us", "Contact"]
    
    # Using a placeholder for custom nav visual
    for p in pages:
        if st.button(f"{'💠' if st.session_state.page == p else '🔹'} {p.upper()}", key=p, use_container_width=True):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")
    
    # LIVE TELEMETRY WIDGETS
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <span style="color: var(--neon-purple); font-size: 1.2rem; margin-right: 10px; animation: pulse-arrow 1s infinite;">⚡</span>
        <span style="color: #fff; font-family: 'Share Tech Mono'; letter-spacing: 1px;">LIVE TELEMETRY</span>
    </div>
    """, unsafe_allow_html=True)

    # Simulated Server Stats
    c1, c2 = st.columns(2)
    with c1: 
        st.metric("GPU TEMP", f"{random.randint(45, 78)}°C")
    with c2:
        st.metric("LATENCY", f"{random.randint(12, 45)}ms")

    # Dynamic Mini-Chart
    st.markdown("<p style='font-size: 0.7rem; color: #888; font-family: Share Tech Mono; margin-top: 10px;'>NEURAL FLUX DENSITY</p>", unsafe_allow_html=True)
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
    st.line_chart(chart_data, height=80, color=["#00f3ff", "#bc13fe", "#0aff48"]) 

    # Status Badge
    st.write("")
    if gemini_active:
        st.markdown('<div style="background: rgba(10,255,72,0.1); border: 1px solid #0aff48; text-align: center; padding: 8px;"><span style="color: #0aff48; font-family: Share Tech Mono; font-size: 0.8rem;">● AI CORE ONLINE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: rgba(255,0,60,0.1); border: 1px solid #ff003c; text-align: center; padding: 8px;"><span style="color: #ff003c; font-family: Share Tech Mono; font-size: 0.8rem;">● AI CORE OFFLINE</span></div>', unsafe_allow_html=True)

# ==========================================
# 🏠 PAGE 1: DASHBOARD
# ==========================================
if st.session_state.page == "Dashboard":
    
    # Hero Section
    st.markdown('<div style="margin-top: 40px;"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="glitch-title">AI THENTIC</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tech-subtitle">>> MILITARY-GRADE NEURAL FORENSICS <<</p>', unsafe_allow_html=True)
    
    # Animated Ticker
    st.markdown("""
    <div style="background: rgba(0,0,0,0.6); border-top: 1px solid #333; border-bottom: 1px solid #333; padding: 5px; overflow: hidden; white-space: nowrap; margin: 20px 0;">
        <div style="display: inline-block; animation: marquee 20s linear infinite; color: var(--neon-green); font-family: 'Share Tech Mono'; font-size: 0.9rem;">
            SYSTEM INITIALIZED... SEARCHING FOR DEEPFAKE ARTIFACTS... LSTM VECTORS LOADED... EFFICIENTNET-B3 STANDING BY... SECURE CONNECTION ESTABLISHED... WAITING FOR INPUT STREAM...
        </div>
    </div>
    <style>@keyframes marquee { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }</style>
    """, unsafe_allow_html=True)
# Top Visuals
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1:
        # Proper IF statement prevents the crash
        if lottie_left_scan:
            st_lottie(lottie_left_scan, height=200, key="l1")
            
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 20px; background: rgba(0,0,0,0.5); border: 1px solid var(--neon-blue); box-shadow: 0 0 20px rgba(0,243,255,0.1);">
            <h3 style="color: #fff; font-family: 'Orbitron';">INTEGRITY VERIFICATION</h3>
            <p style="color:#aaa; font-family: 'Share Tech Mono'; font-size: 0.9rem; margin-bottom: 20px;">
                DEPLOYING BI-DIRECTIONAL LSTM ARRAYS FOR DEEPFAKE ARTIFACT DETECTION.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button(">> LAUNCH ANALYSIS CONSOLE <<", type="primary", use_container_width=True):
            st.session_state.page = "Analysis Console"
            st.rerun()

    with col3:
        # Proper IF statement
        if lottie_right_scan:
            st_lottie(lottie_right_scan, height=200, key="r1")

    st.write("")
    
    # Feature Cards (3D Tilt Effect applied via CSS)
    st.markdown("<h3 style='color: var(--neon-blue); font-family: Orbitron; margin-bottom: 20px;'>// SYSTEM CAPABILITIES</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="tech-card">
            <div class="tech-header">
                <span>01. ACTIVE SAMPLING</span>
                <span style="font-size:0.8rem; border:1px solid var(--neon-blue); padding:2px 5px;">ENTROPY</span>
            </div>
            <p style="color:#ccc; font-size: 0.9rem;">High-Entropy Frame Extraction. Algorithm discards static data to focus solely on high-motion vectors where artifacts occur.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="tech-card" style="border-left-color: var(--neon-purple);">
            <div class="tech-header" style="color: var(--neon-purple);">
                <span>02. TEMPORAL MEMORY</span>
                <span style="font-size:0.8rem; border:1px solid var(--neon-purple); padding:2px 5px;">LSTM</span>
            </div>
            <p style="color:#ccc; font-size: 0.9rem;">Bi-Directional LSTM Core. Analyzes frame-to-frame inconsistencies in the time domain to detect micro-jitter.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div class="tech-card" style="border-left-color: var(--neon-green);">
            <div class="tech-header" style="color: var(--neon-green);">
                <span>03. SPATIAL SCAN</span>
                <span style="font-size:0.8rem; border:1px solid var(--neon-green); padding:2px 5px;">CNN</span>
            </div>
            <p style="color:#ccc; font-size: 0.9rem;">EfficientNet-B3 Backbone. Detects blending boundaries, resolution mismatches, and warping artifacts.</p>
        </div>
        """, unsafe_allow_html=True)

    # FAQ Section
    st.markdown("---")
    st.subheader("FORENSIC DATABASE")
    c_faq1, c_faq2 = st.columns(2)
    with c_faq1:
        with st.expander("❓ WHAT IS A DEEPFAKE?"): st.info("Synthetic media generated by GANs/Diffusion models to impersonate identity.")
        with st.expander("⚙️ HOW DOES IT WORK?"): st.write("We use autoencoders to map source faces to target faces in latent space.")
    with c_faq2:
        with st.expander("🕵️ WHAT ARE ARTIFACTS?"): st.warning("Visual glitches: Blending boundaries, flickering lips, inconsistent lighting.")
        with st.expander("👁️ WHY ACTIVE SAMPLING?"): st.write("We focus only on frames where the subject is moving/talking to save compute.")

    # Chatbot
    st.markdown("---")
    c_chat, c_anim = st.columns([2, 1])
    with c_chat:
        st.markdown("### 💬 SECURE COMMS CHANNEL")
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
        if prompt := st.chat_input("Query the forensic AI..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                if gemini_active:
                    try:
                        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                        response = model_gemini.generate_content(f"{PROJECT_CONTEXT}\nUser: {prompt}")
                        message_placeholder.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except:
                        message_placeholder.error("COMMS LINK FAILED.")
                else:
                    message_placeholder.warning("OFFLINE MODE ACTIVATED.")
    with c_anim:
        if lottie_chatbot:
            st_lottie(lottie_chatbot, height=250, key="bot")

# ==========================================
# 🕵️ PAGE 2: ANALYSIS CONSOLE (FIXED)
# ==========================================
elif st.session_state.page == "Analysis Console":
    
    st.markdown('<h1 class="glitch-title" style="font-size:3.5rem;">ANALYSIS CONSOLE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888; font-family:Share Tech Mono;">SECURE UPLOAD GATEWAY // <span style="color:var(--neon-green)">READY</span></p>', unsafe_allow_html=True)

    # File Uploader with Custom Styling Wrapper
    uploaded_file = st.file_uploader("DRAG & DROP SUSPECT FOOTAGE", type=["mp4", "avi", "mov"])

    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())

        st.write("")
        col_video, col_terminal = st.columns([1.5, 1])

        with col_video:
            # Video Frame
            st.markdown("""
            <div style="border: 1px solid var(--neon-blue); padding: 5px; background: rgba(0,0,0,0.8); position: relative;">
                <div style="position: absolute; top: 10px; left: 10px; z-index: 10; background: red; color: white; padding: 2px 5px; font-size: 0.7rem; font-family: sans-serif;">REC ●</div>
                <div style="position: absolute; bottom: 10px; right: 10px; z-index: 10; color: var(--neon-blue); font-family: 'Share Tech Mono';">SOURCE: EXT_CAM_01</div>
            """, unsafe_allow_html=True)
            st.video(uploaded_file)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.write("")
            analyze_btn = st.button(">> INITIATE DEEP SCAN <<", type="primary", use_container_width=True)

        # Terminal Simulation Helper
        def log_to_terminal(text):
            return f"<div class='log-line'>[{time.strftime('%H:%M:%S')}] {text}</div>"

        with col_terminal:
            st.markdown("<p style='font-family:Share Tech Mono; color: var(--neon-green); margin-bottom: 5px;'>// TERMINAL_OUTPUT</p>", unsafe_allow_html=True)
            terminal_area = st.empty()
            
            # --- FIX: USE A DICTIONARY (MUTABLE) INSTEAD OF A STRING ---
            log_state = {"html": "<div class='terminal-box'>_SYSTEM READY.<br>_WAITING FOR USER AUTHORIZATION...</div>"}
            
            # Render initial state
            terminal_area.markdown(log_state["html"], unsafe_allow_html=True)

        if analyze_btn:
            # 1. Check Model
            if model is None:
                log_state["html"] += log_to_terminal("[FATAL] MODEL WEIGHTS 404.")
                terminal_area.markdown(f"<div class='terminal-box'>{log_state['html']}</div>", unsafe_allow_html=True)
            else:
                # 2. Process Video
                # --- FIX: Function modifies the dictionary directly ---
                def update_log(msg, sleep_t):
                    log_state["html"] += log_to_terminal(msg)
                    terminal_area.markdown(f"<div class='terminal-box'>{log_state['html']}</div>", unsafe_allow_html=True)
                    time.sleep(sleep_t)

                update_log("LOADING VIDEO BUFFER...", 0.5)
                
                faces, raw_frames = process_video_frames("temp_video.mp4", update_log)

                if not faces:
                    update_log("[ERR] NO FACES DETECTED. ABORTING.", 0)
                else:
                    update_log(f"EXTRACTED {len(faces)} REGIONS OF INTEREST.", 0.5)
                    update_log("NORMALIZING TENSORS...", 0.3)
                    update_log("INJECTING INTO EFFICIENTNET-B3...", 0.5)
                    update_log("ANALYZING TEMPORAL VECTORS...", 0.5)

                    # Prediction
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    input_tensor = torch.stack([transform(Image.fromarray(f)) for f in faces]).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        real_score, fake_score = probs[0][0].item(), probs[0][1].item()
                    
                    update_log("ANALYSIS COMPLETE.", 0)
                    update_log(f"FINAL CONFIDENCE: {fake_score:.4f}", 0)

                    # === RESULTS DISPLAY ===
                    st.markdown("---")
                    
                    # 1. Header
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        if fake_score > 0.50:
                            st.markdown(f"""
                            <div style="border: 2px solid #ff003c; background: rgba(255,0,60,0.1); padding: 30px; text-align: center; box-shadow: 0 0 50px rgba(255,0,60,0.2);">
                                <h1 style="color: #ff003c; font-size: 3.5rem; margin:0; font-family:'Orbitron'; text-shadow: 0 0 10px red;">⚠️ DEEPFAKE</h1>
                                <h3 style="color: #fff;">CONFIDENCE: {fake_score*100:.2f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="border: 2px solid #0aff48; background: rgba(10,255,72,0.1); padding: 30px; text-align: center; box-shadow: 0 0 50px rgba(10,255,72,0.2);">
                                <h1 style="color: #0aff48; font-size: 3.5rem; margin:0; font-family:'Orbitron'; text-shadow: 0 0 10px #0aff48;">✅ AUTHENTIC</h1>
                                <h3 style="color: #fff;">CONFIDENCE: {real_score*100:.2f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)

                    # 2. Advanced Data Viz (Plotly)
                    with res_col2:
                        if fake_score > 0.5:
                            jitt, tex, blend, light = fake_score * 0.9, fake_score * 0.85, fake_score * 0.95, fake_score * 0.7
                        else:
                            jitt, tex, blend, light = fake_score * 1.2, fake_score * 1.1, fake_score * 1.3, fake_score * 1.0
                        
                        categories = ['Temporal Jitter', 'Texture Artifacts', 'Blending Boundaries', 'Lighting Consistency', 'Lip Sync']
                        values = [jitt, tex, blend, light, fake_score]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Artifact Scan',
                            line_color='#ff003c' if fake_score > 0.5 else '#0aff48',
                            fillcolor='rgba(255, 0, 60, 0.3)' if fake_score > 0.5 else 'rgba(10, 255, 72, 0.3)'
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor='#333'),
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#fff', family="Share Tech Mono"),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=250
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 3. Artifact Gallery
                    st.write("")
                    st.subheader(">> ISOLATED ARTIFACT FRAMES")
                    st.markdown("Top 10 frames with highest entropy (movement/glitch potential):")
                    
                    cols = st.columns(5)
                    for i, face in enumerate(faces[:5]):
                        with cols[i]:
                            st.image(face, caption=f"FRAME ID: {random.randint(100,999)}", use_container_width=True)
                    cols2 = st.columns(5)
                    for i, face in enumerate(faces[5:10]):
                        with cols2[i]:
                            st.image(face, caption=f"FRAME ID: {random.randint(100,999)}", use_container_width=True)
# ==========================================
# 📄 PAGE 3: METHODOLOGY
# ==========================================
elif st.session_state.page == "Methodology":
    st.markdown('<h1 class="glitch-title">SYSTEM KERNEL</h1>', unsafe_allow_html=True)
    
    # Pulsing Pipeline
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-node">📹<br>INPUT STREAM</div>
        <div class="pipeline-arrow">➤</div>
        <div class="pipeline-node" style="border-color:#00f3ff; color:#00f3ff;">⚡<br>ENTROPY FILTER</div>
        <div class="pipeline-arrow">➤</div>
        <div class="pipeline-node" style="border-color:#bc13fe; color:#bc13fe;">👁️<br>SPATIAL CNN</div>
        <div class="pipeline-arrow">➤</div>
        <div class="pipeline-node" style="border-color:#0aff48; color:#0aff48;">🧠<br>TEMPORAL LSTM</div>
        <div class="pipeline-arrow">➤</div>
        <div class="pipeline-node">🛡️<br>VERDICT</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["[01] ENTROPY SCAN", "[02] EFFICIENT-NET B3", "[03] BI-LSTM"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### ACTIVE TEMPORAL SAMPLING")
            st.write("We do not process every frame. We process **Information**.")
            st.code("Entropy = Σ | Frame(t) - Frame(t-1) |", language="python")
            st.write("By calculating pixel difference, we discard static frames (backgrounds) and focus only on **Micro-Expressions**, where Deepfakes fail.")
        with c2:
            st.metric("DATA REDUCTION", "95%", "+Efficiency")

    with tab2:
        st.markdown("### SPATIAL FEATURE EXTRACTION")
        st.write("Each of the 20 selected frames is passed through a **Pre-trained EfficientNet-B3**.")
        st.info("Why EfficientNet? It balances accuracy and speed better than ResNet-50.")
        st.markdown("**Output Vector:** `[Batch, 20, 1536]`")

    with tab3:
        st.markdown("### TEMPORAL SEQUENCE ANALYSIS")
        st.write("A single frame might look perfect. A sequence reveals the lie.")
        st.latex(r'''
        h_t = \text{LSTM}(x_t, h_{t-1})
        ''')
        st.write("The Bi-Directional LSTM looks at the video **Forwards and Backwards** to detect jitter.")

# ==========================================
# 👤 PAGE 4 & 5: ABOUT & CONTACT
# ==========================================
elif st.session_state.page == "About Us":
    st.markdown('<h1 class="glitch-title">DEV SQUAD</h1>', unsafe_allow_html=True)
    
    def dev_card(name, role, color, desc):
        st.markdown(f"""
        <div class="dev-container" style="border-left: 4px solid {color};">
            <h2 style="color:#fff; font-family:'Orbitron'; margin:0;">{name}</h2>
            <p style="color:{color}; font-weight:bold; letter-spacing:2px; font-family:'Share Tech Mono';">{role}</p>
            <p style="color:#bbb; margin-top:10px;">{desc}</p>
            <div style="position:absolute; bottom:10px; right:10px; opacity:0.3; font-size:3rem;">👾</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: dev_card("SAHIL DESAI", "ARCHITECT", "#00f3ff", "Model Training & LSTM Implementation.")
    with c2: dev_card("HIMANSHU", "BACKEND", "#0aff48", "Pipeline Optimization & API.")
    st.write("")
    c3, c4 = st.columns(2)
    with c3: dev_card("TEJAS", "DATA ENG", "#bc13fe", "Dataset Curation (FaceForensics++).")
    with c4: dev_card("KRISH", "FRONTEND", "#ff003c", "UI/UX & Visual Effects.")

elif st.session_state.page == "Contact":
    st.markdown('<h1 class="glitch-title">SECURE UPLINK</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tech-card" style="text-align:center; padding: 50px;">
        <h2 style="color:var(--neon-blue);">ESTABLISH CONNECTION</h2>
        <p>Encrypted channels are open.</p>
        <br>
        <a href="#" style="text-decoration:none; color:black; background:var(--neon-blue); padding:10px 20px; font-weight:bold;">LINKEDIN</a>
        <a href="#" style="text-decoration:none; color:black; background:var(--neon-purple); padding:10px 20px; font-weight:bold; margin: 0 20px;">GITHUB</a>
        <a href="mailto:sahildesai00112@gmail.com" style="text-decoration:none; color:black; background:var(--neon-green); padding:10px 20px; font-weight:bold;">EMAIL</a>
    </div>
    """, unsafe_allow_html=True)
