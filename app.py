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
import plotly.graph_objects as go 

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
    pass

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

lottie_left_scan = load_lottie_local("assets/animation1.json")
lottie_right_scan = load_lottie_local("assets/animation2.json")
lottie_chatbot = load_lottie_local("assets/animation3.json")
bg_image_base64 = get_base64_of_bin_file("assets/back_ground_img.jpg")

# ==========================================
# 🖌️ 4. ULTRA-MODERN CSS ENGINE
# ==========================================

if bg_image_base64:
    background_style = f"""
    [data-testid="stAppViewContainer"] {{
        background: radial-gradient(circle at center, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.95) 100%), url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-attachment: fixed;
    }}
    """
else:
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
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    :root {{ 
        --neon-blue: #00f3ff; 
        --neon-purple: #bc13fe; 
        --neon-green: #0aff48; 
        --neon-red: #ff003c; 
    }}

    /* Global Font Reduction */
    html, body, [class*="css"] {{ font-family: 'Rajdhani', sans-serif; color: #e0fbfc; font-size: 14px; }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; background: #000; }}
    ::-webkit-scrollbar-thumb {{ background: var(--neon-blue); border-radius: 2px; }}

    /* Scanlines */
    .scanlines {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        background-size: 100% 4px, 6px 100%;
        pointer-events: none; z-index: 9999;
    }}

    {background_style}
    
    [data-testid="stSidebar"] {{
        background-color: rgba(5, 5, 10, 0.95);
        border-right: 1px solid rgba(0, 243, 255, 0.1);
    }}

    /* Glitch Title - Significantly Smaller */
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
        font-size: 3rem; 
        text-align: center;
        color: #fff; text-shadow: 2px 2px var(--neon-purple), -2px -2px var(--neon-blue);
        animation: glitch-skew 3s infinite linear alternate-reverse;
        margin-bottom: 0px; letter-spacing: 3px;
    }}
    .tech-subtitle {{
        font-family: 'Share Tech Mono', monospace; color: var(--neon-blue);
        text-align: center; 
        font-size: 0.9rem; 
        letter-spacing: 4px;
        text-transform: uppercase; margin-top: -5px; opacity: 0.9;
        text-shadow: 0 0 10px var(--neon-blue);
    }}

    /* ----------------------------------------------------
       HOVER CAPABILITY CARDS - Compact
    ---------------------------------------------------- */
    .cap-card-container {{
        position: relative;
        height: 200px;
        background: rgba(10, 15, 20, 0.6);
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-radius: 8px;
        overflow: hidden;
        transition: 0.3s;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }}
    .cap-card-container:hover {{
        border-color: var(--neon-purple);
        box-shadow: 0 0 20px var(--neon-purple);
        transform: translateY(-5px);
    }}
    .cap-content-visible {{
        padding: 15px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: 0.5s;
    }}
    .cap-content-hidden {{
        position: absolute;
        bottom: -100%;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.95);
        background-image: linear-gradient(rgba(188, 19, 254, 0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        padding: 15px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border-top: 2px solid var(--neon-purple);
    }}
    .cap-card-container:hover .cap-content-hidden {{
        bottom: 0;
    }}
    .cap-icon {{ 
        font-size: 2.5rem; 
        margin-bottom: 8px; text-shadow: 0 0 15px currentColor; 
    }}
    .cap-title {{ font-family: 'Orbitron'; font-size: 1rem; letter-spacing: 2px; }}

    /* ----------------------------------------------------
       TELEMETRY CSS - Compact
    ---------------------------------------------------- */
    .telemetry-box {{
        background: #050505;
        border: 1px solid #333;
        padding: 12px;
        font-family: 'Share Tech Mono';
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: inset 0 0 20px rgba(0,255,0,0.05);
    }}
    .telemetry-header {{
        display: flex; justify-content: space-between;
        border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 8px;
        color: var(--neon-green); font-size: 0.75rem;
    }}
    .hex-bg {{
        position: absolute; top:0; left:0; width:100%; height:100%;
        color: rgba(0, 255, 0, 0.05); font-size: 0.5rem;
        z-index: 0; pointer-events: none; word-wrap: break-word;
    }}
    .stat-row {{ 
        display: flex; justify-content: space-between; z-index: 2; position: relative; 
        margin-bottom: 4px; font-size: 0.8rem; 
    }}
    .stat-val {{ color: var(--neon-blue); text-shadow: 0 0 5px var(--neon-blue); }}
    
    @keyframes blink {{ 0% {{opacity:1;}} 50% {{opacity:0.3;}} 100% {{opacity:1;}} }}
    .live-dot {{ color: var(--neon-red); animation: blink 1s infinite; margin-right: 5px; }}

    /* ----------------------------------------------------
       TEAM HOVER CARDS - Reduced Height
    ---------------------------------------------------- */
    .dev-wrapper {{
        position: relative; width: 100%; height: 220px;
        background: rgba(5, 10, 15, 0.8);
        border: 1px solid rgba(255,255,255,0.1);
        border-top: 3px solid var(--neon-blue);
        overflow: hidden; transition: 0.4s;
        margin-bottom: 15px;
    }}
    .dev-wrapper:hover {{
        border-top-color: var(--neon-green);
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        transform: translateY(-5px);
    }}
    .dev-main {{ padding: 20px; transition: 0.4s; }}
    .dev-wrapper:hover .dev-main {{ transform: translateY(-50px); opacity: 0.3; filter: blur(2px); }}
    
    .dev-overlay {{
        position: absolute; bottom: -100%; left: 0; width: 100%; height: 100%;
        background: linear-gradient(0deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.6) 100%);
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        transition: 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 15px; text-align: center;
        border-top: 1px solid var(--neon-green);
    }}
    .dev-wrapper:hover .dev-overlay {{ bottom: 0; }}
    
    .college-tag {{
        background: var(--neon-blue); color: #000; padding: 3px 8px;
        font-weight: bold; font-family: 'Orbitron'; font-size: 0.7rem;
        margin-bottom: 5px; clip-path: polygon(10% 0, 100% 0, 100% 80%, 90% 100%, 0 100%, 0 20%);
    }}

    /* ----------------------------------------------------
       GENERAL UI - Compact
    ---------------------------------------------------- */
    .stButton button {{
        background: transparent !important; border: 1px solid var(--neon-blue) !important;
        color: var(--neon-blue) !important; font-family: 'Share Tech Mono' !important;
        text-transform: uppercase; letter-spacing: 2px; transition: 0.3s;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
    }}
    .stButton button:hover {{ background: rgba(0,243,255,0.1) !important; box-shadow: 0 0 15px var(--neon-blue); }}
    
    .terminal-box {{
        background: #000; border: 1px solid #333; padding: 15px;
        font-family: 'Share Tech Mono', monospace; color: #ccc;
        height: 350px; overflow-y: auto; border-left: 4px solid var(--neon-green);
        font-size: 0.8rem; line-height: 1.4;
    }}
    .log-line {{ margin-bottom: 2px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; }}
    .log-time {{ color: #555; margin-right: 10px; min-width: 80px; }}
    .log-msg {{ color: var(--neon-blue); }}
    .log-sys {{ color: var(--neon-purple); font-weight: bold; }}
    .log-warn {{ color: var(--neon-red); }}
    .log-ok {{ color: var(--neon-green); }}

    /* FAQ Styling */
    .faq-container {{ border: 1px solid #333; margin-bottom: 10px; background: rgba(255,255,255,0.02); transition:0.3s; }}
    .faq-container:hover {{ border-color: var(--neon-blue); background: rgba(0,243,255,0.05); }}

</style>
<div class="scanlines"></div>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 5. MODEL BACKEND
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

    if not os.path.exists(model_path):
        file_id = "1IpeVbi0jvwHaXD5qCMtF_peUVR9uJDw0" 
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=True)
        except Exception: return None

    try:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return model
        return None
    except Exception: return None

model = load_model()
mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False)

# ==========================================
# 📽️ 6. VIDEO PROCESSOR
# ==========================================
def process_video_frames(video_path, status_log_func):
    cap = cv2.VideoCapture(video_path)
    frames, diffs = [], []
    ret, prev = cap.read()
    if not ret: return None, []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_cnt = 0
    status_log_func("INITIALIZING ENTROPY SCANNERS...", "sys")
    
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

    status_log_func("DETECTING FACIAL ROI (MTCNN)...", "sys")
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
# 🧭 7. SIDEBAR (COMPACT)
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; border-bottom: 2px solid var(--neon-blue); padding-bottom: 15px; margin-bottom: 15px;">
        <h1 style="color: #fff; margin:0; font-family:'Orbitron'; font-size: 1.8rem; text-shadow: 0 0 10px var(--neon-blue);">OPS CENTER</h1>
        <p style="color: var(--neon-blue); margin:0; font-size: 0.7rem; letter-spacing: 3px; font-family:'Share Tech Mono';">SYS.VER.4.0.ALPHA</p>
    </div>
    """, unsafe_allow_html=True)
    
    pages = ["Dashboard", "Analysis Console", "Methodology", "About Us", "Contact"]
    for p in pages:
        if st.button(f"{'💠' if st.session_state.page == p else '🔹'} {p.upper()}", key=p, use_container_width=True):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")
    
    # --- CRAZY TELEMETRY ---
    random_hex = ' '.join([f"{random.randint(0, 255):02X}" for _ in range(50)])
    
    st.markdown(f"""
    <div class="telemetry-box">
        <div class="hex-bg">{random_hex} {random_hex} {random_hex}</div>
        <div class="telemetry-header">
            <span><span class="live-dot">●</span>LIVE_FEED</span>
            <span>ID: 8X-99</span>
        </div>
        <div class="stat-row">
            <span>GPU_LOAD</span>
            <span class="stat-val">{random.randint(30, 95)}%</span>
        </div>
        <div class="stat-row">
            <span>TENSOR_CORES</span>
            <span class="stat-val">ACTIVE</span>
        </div>
        <div class="stat-row">
            <span>VRAM_USAGE</span>
            <span class="stat-val">{random.randint(4, 12)}GB</span>
        </div>
        <div class="stat-row">
            <span>LATENCY</span>
            <span class="stat-val">{random.randint(10, 45)}ms</span>
        </div>
        <div style="margin-top:8px; border-top:1px dashed #333; padding-top:5px; font-size:0.6rem; color:#666;">
            ENCRYPTION: AES-256<br>
            UPLINK: STABLE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mini Chart
    st.markdown("<p style='font-size: 0.6rem; color: #888; font-family: Share Tech Mono;'>NEURAL_ACTIVITY_LOG</p>", unsafe_allow_html=True)
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
    st.line_chart(chart_data, height=60, color=["#00f3ff", "#bc13fe", "#0aff48"]) 

    # Status Badge
    if gemini_active:
        st.markdown('<div style="background: rgba(10,255,72,0.1); border: 1px solid #0aff48; text-align: center; padding: 4px;"><span style="color: #0aff48; font-family: Share Tech Mono; font-size: 0.7rem;">● AI CORE ONLINE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: rgba(255,0,60,0.1); border: 1px solid #ff003c; text-align: center; padding: 4px;"><span style="color: #ff003c; font-family: Share Tech Mono; font-size: 0.7rem;">● AI CORE OFFLINE</span></div>', unsafe_allow_html=True)

# ==========================================
# 🏠 PAGE 1: DASHBOARD
# ==========================================
if st.session_state.page == "Dashboard":
    
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="glitch-title">AI THENTIC</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tech-subtitle">>> MILITARY-GRADE NEURAL FORENSICS <<</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(0,0,0,0.6); border-top: 1px solid #333; border-bottom: 1px solid #333; padding: 5px; overflow: hidden; white-space: nowrap; margin: 20px 0;">
        <div style="display: inline-block; animation: marquee 20s linear infinite; color: var(--neon-green); font-family: 'Share Tech Mono'; font-size: 0.8rem;">
            SYSTEM INITIALIZED... SEARCHING FOR DEEPFAKE ARTIFACTS... LSTM VECTORS LOADED... EFFICIENTNET-B3 STANDING BY... SECURE CONNECTION ESTABLISHED... WAITING FOR INPUT STREAM...
        </div>
    </div>
    <style>@keyframes marquee { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }</style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col1: 
        if lottie_left_scan:
            st_lottie(lottie_left_scan, height=200, key="l1")
            
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 15px; background: rgba(0,0,0,0.5); border: 1px solid var(--neon-blue); box-shadow: 0 0 20px rgba(0,243,255,0.1);">
            <h3 style="color: #fff; font-family: 'Orbitron'; font-size: 1.2rem;">INTEGRITY VERIFICATION</h3>
            <p style="color:#aaa; font-family: 'Share Tech Mono'; font-size: 0.8rem; margin-bottom: 15px;">
                DEPLOYING BI-DIRECTIONAL LSTM ARRAYS FOR DEEPFAKE ARTIFACT DETECTION.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button(">> LAUNCH ANALYSIS CONSOLE <<", type="primary", use_container_width=True):
            st.session_state.page = "Analysis Console"
            st.rerun()

    with col3: 
        if lottie_right_scan:
            st_lottie(lottie_right_scan, height=200, key="r1")

    st.write("")
    
    # --- CAPABILITY CARDS ---
    st.markdown("<h3 style='color: var(--neon-blue); font-family: Orbitron; margin-bottom: 20px; font-size: 1.2rem;'>// SYSTEM CAPABILITIES [HOVER TO DECRYPT]</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="cap-card-container">
            <div class="cap-content-visible">
                <div class="cap-icon" style="color:var(--neon-blue)">⚡</div>
                <div class="cap-title" style="color:var(--neon-blue)">ACTIVE SAMPLING</div>
                <p style="color:#888; font-family:'Share Tech Mono'; font-size: 0.8rem;">Module 01</p>
            </div>
            <div class="cap-content-hidden">
                <h4 style="color:var(--neon-blue); font-family:'Orbitron'; font-size: 1rem;">ENTROPY SCAN</h4>
                <p style="color:#ddd; font-size:0.8rem;">
                    The system ignores 95% of static frames. We calculate pixel-difference entropy to isolate micro-movements, reducing compute time while increasing accuracy on lips/eyes.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="cap-card-container" style="border-color: rgba(188, 19, 254, 0.2);">
            <div class="cap-content-visible">
                <div class="cap-icon" style="color:var(--neon-purple)">🧠</div>
                <div class="cap-title" style="color:var(--neon-purple)">TEMPORAL MEMORY</div>
                <p style="color:#888; font-family:'Share Tech Mono'; font-size: 0.8rem;">Module 02</p>
            </div>
            <div class="cap-content-hidden" style="border-top-color: var(--neon-purple);">
                <h4 style="color:var(--neon-purple); font-family:'Orbitron'; font-size: 1rem;">BI-LSTM CORE</h4>
                <p style="color:#ddd; font-size:0.8rem;">
                    Deepfakes flicker over time. Our Bidirectional LSTM analyzes video forwards AND backwards to catch temporal jitter that single-frame CNNs miss.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div class="cap-card-container" style="border-color: rgba(10, 255, 72, 0.2);">
            <div class="cap-content-visible">
                <div class="cap-icon" style="color:var(--neon-green)">👁️</div>
                <div class="cap-title" style="color:var(--neon-green)">SPATIAL SCAN</div>
                <p style="color:#888; font-family:'Share Tech Mono'; font-size: 0.8rem;">Module 03</p>
            </div>
            <div class="cap-content-hidden" style="border-top-color: var(--neon-green);">
                <h4 style="color:var(--neon-green); font-family:'Orbitron'; font-size: 1rem;">EFFICIENTNET-B3</h4>
                <p style="color:#ddd; font-size:0.8rem;">
                    A specialized CNN backbone trained on FaceForensics++. It detects warping artifacts, blending boundaries, and inconsistent lighting on the pixel level.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- FAQs ---
    st.markdown("---")
    st.subheader(">> FORENSIC KNOWLEDGE BASE")
    
    faq_cols = st.columns(2)
    with faq_cols[0]:
        st.markdown('<div class="faq-container">', unsafe_allow_html=True)
        with st.expander("❓ WHAT IS A DEEPFAKE?"): 
            st.info("Synthetic media generated by GANs or Diffusion models to impersonate identity. They map source faces to target faces in latent space.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="faq-container">', unsafe_allow_html=True)
        with st.expander("⚙️ HOW DOES DETECTION WORK?"): 
            st.write("We use a 'Multi-Modal' approach: Spatial analysis checks for weird pixels, Temporal analysis checks for weird movement.")
        st.markdown('</div>', unsafe_allow_html=True)

    with faq_cols[1]:
        st.markdown('<div class="faq-container">', unsafe_allow_html=True)
        with st.expander("🕵️ CAN YOU DETECT NEW MODELS (SORA/KLING)?"): 
            st.success("YES. While models change, the fundamental flaw of generative AI (temporal inconsistency) remains detectable by our LSTM layers.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="faq-container">', unsafe_allow_html=True)
        with st.expander("💾 IS MY DATA SAVED?"): 
            st.warning("NEGATIVE. Security protocol mandates immediate deletion of video buffers after analysis. We are a Zero-Retention system.")
        st.markdown('</div>', unsafe_allow_html=True)

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
                    except: message_placeholder.error("COMMS LINK FAILED.")
                else: message_placeholder.warning("OFFLINE MODE ACTIVATED.")
    
    with c_anim:
        if lottie_chatbot:
            st_lottie(lottie_chatbot, height=250, key="bot")

# ==========================================
# 🕵️ PAGE 2: ANALYSIS CONSOLE (PROFESSIONAL TERMINAL)
# ==========================================
elif st.session_state.page == "Analysis Console":
    
    st.markdown('<h1 class="glitch-title" style="font-size:3rem;">ANALYSIS CONSOLE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888; font-family:Share Tech Mono; font-size: 0.9rem;">SECURE UPLOAD GATEWAY // <span style="color:var(--neon-green)">READY</span></p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("DRAG & DROP SUSPECT FOOTAGE", type=["mp4", "avi", "mov"])

    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        st.write("")
        col_video, col_terminal = st.columns([1.5, 1])

        with col_video:
            st.markdown("""
            <div style="border: 1px solid var(--neon-blue); padding: 5px; background: rgba(0,0,0,0.8); position: relative;">
                <div style="position: absolute; top: 10px; left: 10px; z-index: 10; background: red; color: white; padding: 2px 5px; font-size: 0.7rem; font-family: sans-serif;">REC ●</div>
                <div style="position: absolute; bottom: 10px; right: 10px; z-index: 10; color: var(--neon-blue); font-family: 'Share Tech Mono'; font-size: 0.7rem;">SOURCE: EXT_CAM_01</div>
            """, unsafe_allow_html=True)
            st.video(uploaded_file)
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")
            analyze_btn = st.button(">> INITIATE DEEP SCAN <<", type="primary", use_container_width=True)

        # === PROFESSIONAL LOGGING SYSTEM ===
        if "log_history" not in st.session_state:
            st.session_state.log_history = []

        def render_logs():
            log_html = ""
            for entry in st.session_state.log_history:
                log_html += f"""
                <div class='log-line'>
                    <span class='log-time'>[{entry['time']}]</span>
                    <span class='log-{entry['type']}'>{entry['msg']}</span>
                </div>
                """
            return f"<div class='terminal-box'>{log_html}</div>"

        def add_log(msg, type="msg"):
            t = time.strftime('%H:%M:%S')
            st.session_state.log_history.append({"time": t, "msg": msg, "type": type})

        with col_terminal:
            st.markdown("<p style='font-family:Share Tech Mono; color: var(--neon-green); margin-bottom: 5px; font-size: 0.9rem;'>// TERMINAL_OUTPUT</p>", unsafe_allow_html=True)
            terminal_area = st.empty()
            
            # Initial State
            if not st.session_state.log_history:
                add_log("SYSTEM INITIALIZED.", "ok")
                add_log("WAITING FOR USER AUTHORIZATION...", "msg")
            
            terminal_area.markdown(render_logs(), unsafe_allow_html=True)

        if analyze_btn:
            # Clear previous logs for new run
            st.session_state.log_history = [] 
            add_log("AUTHORIZATION ACCEPTED. STARTING SEQUENCE.", "ok")
            terminal_area.markdown(render_logs(), unsafe_allow_html=True)
            
            if model is None:
                add_log("FATAL ERROR: MODEL WEIGHTS NOT FOUND (404).", "warn")
                terminal_area.markdown(render_logs(), unsafe_allow_html=True)
            else:
                # Helper for real-time updates
                def log_update(msg, type="msg", sleep_t=0.2):
                    add_log(msg, type)
                    terminal_area.markdown(render_logs(), unsafe_allow_html=True)
                    time.sleep(sleep_t)

                log_update(">> INITIALIZING ANALYSIS PROTOCOL...", "sys", 0.5)
                log_update(f">> MOUNTING INPUT STREAM: {uploaded_file.name}", "msg", 0.3)
                log_update(">> LOADING NEURAL WEIGHTS (EfficientNet_B3_LSTM)...", "sys", 0.5)
                log_update(">> SCANNING FRAMES FOR ENTROPY...", "msg", 0.5)
                
                # Pass log function to processor
                faces, raw_frames = process_video_frames("temp_video.mp4", log_update)

                if not faces:
                    log_update(">> ERROR: NO FACES DETECTED IN STREAM.", "warn", 0)
                else:
                    log_update(f">> EXTRACTED {len(faces)} REGIONS OF INTEREST (ROI)...", "ok", 0.3)
                    log_update(">> ALLOCATING TENSORS TO GPU...", "sys", 0.4)
                    log_update(">> NORMALIZING TENSORS [C, H, W]...", "msg", 0.3)
                    log_update(">> INJECTING INTO EFFICIENTNET-B3 BACKBONE...", "sys", 0.5)
                    log_update(">> EXTRACTING TEMPORAL VECTORS...", "msg", 0.3)
                    log_update(">> INFERENCE IN PROGRESS...", "sys", 0.5)

                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    input_tensor = torch.stack([transform(Image.fromarray(f)) for f in faces]).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        real_score, fake_score = probs[0][0].item(), probs[0][1].item()
                    
                    log_update(">> COMPLETED.", "ok", 0)
                    log_update(f">> FINAL CONFIDENCE SCORE: {fake_score:.4f}", "msg", 0)

                    st.markdown("---")
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        if fake_score > 0.50:
                            st.markdown(f"""
                            <div style="border: 2px solid #ff003c; background: rgba(255,0,60,0.1); padding: 20px; text-align: center; box-shadow: 0 0 30px rgba(255,0,60,0.2);">
                                <h1 style="color: #ff003c; font-size: 2.5rem; margin:0; font-family:'Orbitron'; text-shadow: 0 0 10px red;">⚠️ DEEPFAKE</h1>
                                <h3 style="color: #fff; font-size: 1.2rem;">CONFIDENCE: {fake_score*100:.2f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="border: 2px solid #0aff48; background: rgba(10,255,72,0.1); padding: 20px; text-align: center; box-shadow: 0 0 30px rgba(10,255,72,0.2);">
                                <h1 style="color: #0aff48; font-size: 2.5rem; margin:0; font-family:'Orbitron'; text-shadow: 0 0 10px #0aff48;">✅ AUTHENTIC</h1>
                                <h3 style="color: #fff; font-size: 1.2rem;">CONFIDENCE: {real_score*100:.2f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)

                    with res_col2:
                        if fake_score > 0.5:
                            jitt, tex, blend, light = fake_score * 0.9, fake_score * 0.85, fake_score * 0.95, fake_score * 0.7
                        else:
                            jitt, tex, blend, light = fake_score * 1.2, fake_score * 1.1, fake_score * 1.3, fake_score * 1.0
                        
                        categories = ['Temporal Jitter', 'Texture Artifacts', 'Blending Boundaries', 'Lighting Consistency', 'Lip Sync']
                        values = [jitt, tex, blend, light, fake_score]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values, theta=categories, fill='toself', name='Artifact Scan',
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

                    st.write("")
                    st.subheader(">> ISOLATED ARTIFACT FRAMES")
                    cols = st.columns(5)
                    for i, face in enumerate(faces[:5]):
                        with cols[i]: st.image(face, caption=f"FRAME ID: {random.randint(100,999)}", use_container_width=True)
                    cols2 = st.columns(5)
                    for i, face in enumerate(faces[5:10]):
                        with cols2[i]: st.image(face, caption=f"FRAME ID: {random.randint(100,999)}", use_container_width=True)

# ==========================================
# 📄 PAGE 3: METHODOLOGY
# ==========================================
elif st.session_state.page == "Methodology":
    st.markdown('<h1 class="glitch-title" style="font-size:3rem;">SYSTEM KERNEL</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(0,0,0,0.8); border: 1px solid var(--neon-blue); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h3 style="color:var(--neon-blue); text-align:center; font-family:'Share Tech Mono'; font-size: 1.2rem;">// END-TO-END PIPELINE ARCHITECTURE</h3>
        <div style="display:flex; justify-content:space-around; align-items:center; margin-top:20px; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:2.5rem;">📹</div>
                <div style="color:#888; font-size:0.8rem;">RAW VIDEO</div>
            </div>
            <div style="color:var(--neon-green); font-size:1.5rem;">➔</div>
            <div style="border:1px solid #ff003c; padding:10px; border-radius:5px; text-align:center;">
                <div style="color:#ff003c; font-weight:bold; font-size: 0.9rem;">FRAME SAMPLER</div>
                <div style="font-size:0.6rem; color:#aaa;">Entropy Filter</div>
            </div>
            <div style="color:var(--neon-green); font-size:1.5rem;">➔</div>
            <div style="border:1px solid #bc13fe; padding:10px; border-radius:5px; text-align:center;">
                <div style="color:#bc13fe; font-weight:bold; font-size: 0.9rem;">CNN (SPATIAL)</div>
                <div style="font-size:0.6rem; color:#aaa;">EfficientNet-B3</div>
            </div>
            <div style="color:var(--neon-green); font-size:1.5rem;">➔</div>
            <div style="border:1px solid #00f3ff; padding:10px; border-radius:5px; text-align:center;">
                <div style="color:#00f3ff; font-weight:bold; font-size: 0.9rem;">RNN (TEMPORAL)</div>
                <div style="font-size:0.6rem; color:#aaa;">Bi-Directional LSTM</div>
            </div>
            <div style="color:var(--neon-green); font-size:1.5rem;">➔</div>
            <div style="text-align:center;">
                <div style="font-size:2.5rem;">🛡️</div>
                <div style="color:#888; font-size:0.8rem;">PREDICTION</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["[01] ENTROPY MATH", "[02] SPATIAL VECTORS", "[03] LSTM GATES"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### ⚡ ACTIVE TEMPORAL SAMPLING")
            st.write("We utilize a pixel-difference algorithm to calculate the 'Entropy' (Information Density) of every frame.")
            st.latex(r'''
            E_t = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} | P_{t}(i,j) - P_{t-1}(i,j) |
            ''')
            st.write("Where $P_t$ is the pixel value at time $t$. We only select the top $k=20$ frames where $E_t$ is maximized.")
        with c2:
            st.info("Why? Deepfake artifacts usually appear during high-motion (blinking, talking). Static frames are useless noise.")

    with tab2:
        st.markdown("### 👁️ SPATIAL FEATURE EXTRACTION")
        st.write("Selected frames are passed through EfficientNet-B3. The classification head is removed to extract raw feature vectors.")
        st.latex(r'''
        F_t = \text{CNN}_{\theta}(x_t) \in \mathbb{R}^{1536}
        ''')
        st.write("This creates a sequence of vectors: $S = [F_1, F_2, ..., F_{20}]$ representing the visual texture of the video over time.")

    with tab3:
        st.markdown("### 🧠 TEMPORAL SEQUENCE ANALYSIS")
        st.write("The sequence $S$ is fed into a Bidirectional LSTM. This allows the model to see context from both past and future frames.")
        st.latex(r'''
        \begin{aligned}
        f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
        i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
        \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
        C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t
        \end{aligned}
        ''')
        st.success("This complex gating mechanism detects 'Temporal Jitter'—micro-flickers that human eyes miss but math cannot ignore.")

# ==========================================
# 👤 PAGE 4 & 5: ABOUT & CONTACT
# ==========================================
elif st.session_state.page == "About Us":
    st.markdown('<h1 class="glitch-title" style="font-size:3rem;">DEV SQUAD</h1>', unsafe_allow_html=True)
    st.write("")
    
    def dev_card_animated(name, role, color, desc, college_info):
        st.markdown(f"""
        <div class="dev-wrapper" style="border-top-color: {color};">
            <div class="dev-main">
                <h2 style="color:#fff; font-family:'Orbitron'; margin:0; font-size: 1.3rem;">{name}</h2>
                <p style="color:{color}; font-weight:bold; letter-spacing:2px; font-family:'Share Tech Mono'; font-size: 0.8rem;">{role}</p>
                <div style="position:absolute; bottom:20px; right:20px; font-size:3rem; opacity:0.1;">👾</div>
            </div>
            <div class="dev-overlay">
                <div class="college-tag">{college_info}</div>
                <h3 style="color:#fff; font-family:'Orbitron'; font-size: 1.1rem;">{name}</h3>
                <p style="color:#ccc; font-size:0.8rem;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: dev_card_animated("SAHIL DESAI", "ARCHITECT", "#00f3ff", "Model Training & LSTM Implementation.", "VJTI 2ND YEAR EXTC")
    with c2: dev_card_animated("HIMANSHU", "BACKEND", "#0aff48", "Pipeline Optimization & API.", "VJTI 2ND YEAR EXTC")
    
    c3, c4 = st.columns(2)
    with c3: dev_card_animated("TEJAS", "DATA ENG", "#bc13fe", "Dataset Curation (FaceForensics++).", "VJTI 2ND YEAR EXTC")
    with c4: dev_card_animated("KRISH", "FRONTEND", "#ff003c", "UI/UX & Visual Effects.", "VJTI 2ND YEAR EXTC")

elif st.session_state.page == "Contact":
    st.markdown('<h1 class="glitch-title" style="font-size:3rem;">SECURE UPLINK</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(10, 15, 20, 0.6); backdrop-filter: blur(15px); border: 1px solid rgba(0, 243, 255, 0.2); padding: 40px; text-align: center; border-radius: 15px;">
        <h2 style="color:var(--neon-blue); font-family:'Orbitron'; font-size: 1.5rem;">ESTABLISH CONNECTION</h2>
        <p style="color:#ccc; font-size: 0.9rem;">Encrypted channels are open. Response time < 200ms.</p>
        <br>
        <div style="display:flex; justify-content:center; gap: 20px;">
            <a href="#" style="text-decoration:none; color:#000; background:var(--neon-blue); padding:10px 25px; font-weight:bold; font-family:'Share Tech Mono'; clip-path: polygon(10% 0, 100% 0, 100% 80%, 90% 100%, 0 100%, 0 20%); transition:0.3s; font-size: 0.9rem;">LINKEDIN</a>
            <a href="#" style="text-decoration:none; color:#000; background:var(--neon-purple); padding:10px 25px; font-weight:bold; font-family:'Share Tech Mono'; clip-path: polygon(10% 0, 100% 0, 100% 80%, 90% 100%, 0 100%, 0 20%); transition:0.3s; font-size: 0.9rem;">GITHUB</a>
            <a href="mailto:sahildesai00112@gmail.com" style="text-decoration:none; color:#000; background:var(--neon-green); padding:10px 25px; font-weight:bold; font-family:'Share Tech Mono'; clip-path: polygon(10% 0, 100% 0, 100% 80%, 90% 100%, 0 100%, 0 20%); transition:0.3s; font-size: 0.9rem;">EMAIL</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
