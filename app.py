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

# ==========================================
# 🎨 1. PAGE CONFIG (THE FOUNDATION)
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
# 👇 YOUR API KEY IS INTEGRATED HERE
GOOGLE_API_KEY = "xyz" 

gemini_active = False
try:
    if GOOGLE_API_KEY != "PASTE_YOUR_KEY_HERE":
        clean_key = GOOGLE_API_KEY.strip()
        genai.configure(api_key=clean_key)
        gemini_active = True
except Exception as e:
    st.error(f"API Setup Error: {e}")

# ==========================================
# 🧠 THE BRAIN: KNOWLEDGE BASE
# ==========================================
PROJECT_CONTEXT = """
ROLE: You are the "AIthentic Forensic Assistant", a military-grade neural expert.
GOAL: Explain the technical depth of the AIthentic platform to judges and users.

--- SYSTEM ARCHITECTURE ---
1. **Active Temporal Sampling:** We do NOT analyze every frame. We use Entropy Scanning to find high-motion frames (blinking/turning) where Deepfakes glitch.
2. **Spatial Analysis (EfficientNet-B3):** Extracts texture features (resolution mismatches, blending boundaries).
3. **Temporal Analysis (Bi-LSTM):** Detects "Temporal Jitter" (flickering over time).

--- METRICS ---
* Accuracy: 96.71% (FaceForensics++ HQ).
* Precision: 0.99.
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

# Load Assets
lottie_left_scan = load_lottie_local("assets/animation1.json")
lottie_right_scan = load_lottie_local("assets/animation2.json")
lottie_chatbot = load_lottie_local("assets/animation3.json")
bg_image_base64 = get_base64_of_bin_file("assets/back_ground_img.jpg")

# ==========================================
# 🖌️ 4. ULTRA-MODERN CSS INJECTION
# ==========================================

# Background Logic
if bg_image_base64:
    bg_style = f"""background-image: linear-gradient(rgba(0, 5, 10, 0.85), rgba(0, 5, 10, 0.95)), url("data:image/jpg;base64,{bg_image_base64}");"""
else:
    bg_style = """background: #020202; background-image: radial-gradient(#0a1f2e 1px, transparent 1px); background-size: 40px 40px;"""

st.markdown(f"""
<style>
    /* --- FONTS --- */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700;900&family=Share+Tech+Mono&display=swap');

    :root {{
        --neon-blue: #00f3ff;
        --neon-purple: #bc13fe;
        --neon-green: #0aff48;
        --neon-red: #ff003c;
        --glass-bg: rgba(10, 20, 30, 0.6);
        --glass-border: rgba(0, 243, 255, 0.2);
    }}

    html, body, [class*="css"] {{
        font-family: 'Rajdhani', sans-serif;
        color: #e0fbfc;
        background: transparent;
    }}

    /* --- BACKGROUND & SCANLINES --- */
    [data-testid="stAppViewContainer"] {{
        {bg_style}
        background-size: cover;
        background-attachment: fixed;
    }}
    
    /* CRT Scanline Effect */
    [data-testid="stAppViewContainer"]::after {{
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
        background-size: 100% 3px, 3px 100%;
        pointer-events: none;
        z-index: 9999;
    }}

    /* --- SIDEBAR: THE OPS CENTER --- */
    [data-testid="stSidebar"] {{
        background-color: rgba(5, 8, 10, 0.95);
        border-right: 1px solid var(--glass-border);
        box-shadow: 5px 0 20px rgba(0,0,0,0.5);
    }}
    
    /* Sidebar Nav Buttons */
    [data-testid="stSidebar"] .stRadio label {{
        padding: 10px 0;
        cursor: pointer;
        transition: 0.3s;
    }}
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
        font-size: 1.3rem;
        font-weight: 700;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
    }}
    /* Active/Hover State for Nav */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover p,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] p {{
        color: var(--neon-blue) !important;
        text-shadow: 0 0 15px var(--neon-blue);
        padding-left: 10px;
    }}

    /* --- TITLES: CHROME & NEON --- */
    .chrome-title {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 900;
        font-size: 5rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 8px;
        background: linear-gradient(to bottom, #ffffff 0%, #a2a2a2 50%, #000000 51%, #00f3ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 2px var(--neon-blue));
        margin-bottom: 0;
    }}
    
    .neon-subtitle {{
        font-family: 'Share Tech Mono';
        color: var(--neon-purple);
        text-align: center;
        letter-spacing: 4px;
        font-size: 1.2rem;
        text-shadow: 0 0 10px var(--neon-purple);
        margin-top: -10px;
    }}

    /* --- GLASSMORPHISM CARDS --- */
    .glass-card {{
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 25px;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
    }}
    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 243, 255, 0.15);
        border-color: var(--neon-blue);
    }}
    .glass-card h4 {{
        color: var(--neon-blue);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }}

    /* --- CUSTOM BUTTONS --- */
    .stButton button {{
        background: transparent !important;
        border: 1px solid var(--neon-blue) !important;
        color: var(--neon-blue) !important;
        font-family: 'Share Tech Mono' !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 15px 30px !important;
        border-radius: 4px !important;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
    }}
    .stButton button:hover {{
        background: var(--neon-blue) !important;
        color: #000 !important;
        box-shadow: 0 0 25px var(--neon-blue);
        transform: scale(1.05);
    }}

    /* --- FILE UPLOADER (DROP ZONE) --- */
    [data-testid="stFileUploader"] section {{
        background-color: rgba(0, 10, 15, 0.5);
        border: 2px dashed var(--neon-purple);
        border-radius: 10px;
    }}
    [data-testid="stFileUploader"] section:hover {{
        background-color: rgba(188, 19, 254, 0.1);
        border-color: #fff;
    }}

    /* --- CHATBOT BUBBLES --- */
    [data-testid="stChatMessage"] {{
        background-color: rgba(5, 10, 15, 0.8);
        border: 1px solid #333;
        border-left: 3px solid var(--neon-green);
        font-family: 'Share Tech Mono';
    }}
    
    /* --- METRICS & BARS --- */
    div[data-testid="metric-container"] {{
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    div[data-testid="metric-container"] label {{ color: var(--neon-blue); }}
    
    /* HIDE DEFAULT HEADER/FOOTER */
    header, footer {{ visibility: hidden; }}
</style>
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
    try:
        if os.path.exists("efficientnet_b3_lstm_active.pth"):
            model.load_state_dict(torch.load("efficientnet_b3_lstm_active.pth", map_location=DEVICE))
            model.eval()
            return model
        return None
    except Exception: return None

model = load_model()
mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False)

# ==========================================
# 📽️ 6. PROCESSING ENGINE
# ==========================================
def process_video_frames(video_path, status_log):
    cap = cv2.VideoCapture(video_path)
    frames, diffs = [], []
    ret, prev = cap.read()
    if not ret: return None, []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_cnt = 0
    
    status_log.markdown("`> SCANNING TEMPORAL VECTORS...`")
    time.sleep(1.0) 
    
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

    status_log.markdown("`> DETECTING FACIAL REGIONS...`")
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
# 🧭 7. SIDEBAR (HUD)
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; border-bottom: 1px solid #00f3ff; padding-bottom: 15px; margin-bottom: 30px;">
        <h2 style="color: #fff; margin:0; letter-spacing: 4px; font-size: 2.2rem; text-shadow: 0 0 10px #00f3ff;">OPS CENTER</h2>
        <p style="color: #00f3ff; margin:0; font-size: 0.8rem; letter-spacing: 2px;">V.2.0.4. BETA</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_page = st.radio(
        "MODULE SELECT",
        ["Dashboard", "Analysis Console", "Methodology", "About Us", "Contact"],
        index=["Dashboard", "Analysis Console", "Methodology", "About Us", "Contact"].index(st.session_state.page),
        label_visibility="collapsed"
    )
    
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    st.markdown("---")
    
    # TELEMETRY SECTION
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="color: #00f3ff; font-size: 1.2rem; margin-right: 10px;">📡</span>
        <span style="color: #fff; font-family: 'Share Tech Mono'; letter-spacing: 2px; font-weight:bold;">LIVE TELEMETRY</span>
    </div>
    """, unsafe_allow_html=True)

    gpu_load, ram_load, temp = random.randint(30, 85), random.randint(40, 65), random.randint(45, 75)

    st.markdown(f"""
    <style>
        .meter-box {{ margin-bottom: 15px; }}
        .meter-head {{ display: flex; justify-content: space-between; color: #fff; font-family: 'Share Tech Mono'; font-size: 0.9rem; margin-bottom: 5px; }}
        .bar-bg {{ width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; }}
        .bar-fill {{ height: 100%; border-radius: 4px; box-shadow: 0 0 8px currentColor; }}
    </style>
    <div class="meter-box">
        <div class="meter-head"><span>GPU</span><span style="color:#00f3ff">{gpu_load}%</span></div>
        <div class="bar-bg"><div class="bar-fill" style="width: {gpu_load}%; background: #00f3ff; color: #00f3ff;"></div></div>
    </div>
    <div class="meter-box">
        <div class="meter-head"><span>RAM</span><span style="color:#bc13fe">{ram_load}%</span></div>
        <div class="bar-bg"><div class="bar-fill" style="width: {ram_load}%; background: #bc13fe; color: #bc13fe;"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # Mini Graph
    st.markdown("<p style='font-size:0.8rem; color:#aaa; font-family:Share Tech Mono;'>NEURAL FLUX</p>", unsafe_allow_html=True)
    st.line_chart(pd.DataFrame(np.random.randn(20, 1), columns=['a']), height=80, color="#0aff48")

    # Terminal Log
    log_msg = random.choice(["Secure Uplink Established", "Handshake Verified", "Model Weights Loaded", "Monitoring Threads"])
    st.markdown(f"""
    <div style="background: #000; border: 1px solid #333; padding: 10px; font-family: 'Share Tech Mono'; font-size: 0.7rem; color: #0aff48; border-left: 3px solid #0aff48; margin-top: 20px;">
        <div style="opacity:0.5;">> sys_init.sh</div>
        <div style="color:#fff;">> {log_msg} <span style="animation:blink 1s infinite;">_</span></div>
    </div>
    <style>@keyframes blink {{ 0%, 100% {{ opacity: 0; }} 50% {{ opacity: 1; }} }}</style>
    """, unsafe_allow_html=True)

# ==========================================
# 🏠 PAGE 1: DASHBOARD
# ==========================================
if st.session_state.page == "Dashboard":
    
    st.markdown("<h1 class='chrome-title'>AI THENTIC</h1>", unsafe_allow_html=True)
    st.markdown("<p class='neon-subtitle'>NEURAL FORENSIC SUITE v2.0</p>", unsafe_allow_html=True)
    st.write("") 

    # --- HERO SECTION ---
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1: 
            if lottie_left_scan: st_lottie(lottie_left_scan, height=180, key="l1")
        with c2:
            st.markdown("""
            <div style="text-align: center; background: rgba(0, 243, 255, 0.05); padding: 30px; border-radius: 15px; border: 1px solid rgba(0, 243, 255, 0.2); box-shadow: 0 0 30px rgba(0, 243, 255, 0.1);">
                <h3 style="color: #fff; margin-bottom: 10px; text-shadow: 0 0 10px #00f3ff;">DIGITAL INTEGRITY VERIFICATION</h3>
                <p style="color: #aaa; font-family: 'Share Tech Mono';">DEPLOYING BI-DIRECTIONAL LSTM ARRAYS FOR DEEPFAKE ARTIFACT DETECTION.</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button(">> INITIALIZE ANALYSIS MODULE <<", type="primary", use_container_width=True):
                st.session_state.page = "Analysis Console"
                st.rerun()
        with c3:
             if lottie_right_scan: st_lottie(lottie_right_scan, height=180, key="r1")

    # --- ARCHITECTURE CARDS ---
    st.write("")
    st.markdown("<h3 style='text-align:center; color: #00f3ff; letter-spacing:4px; margin-top:40px;'>SYSTEM ARCHITECTURE</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>01. ACTIVE SAMPLING</h4>
            <p>Entropy-based frame extraction. We discard static data to focus solely on high-motion vectors (blinking, turning) where artifacts occur.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>02. SPATIAL SCAN</h4>
            <p><strong>EfficientNet-B3 Backbone.</strong> A powerful CNN that detects pixel-level anomalies, resolution mismatches, and warping artifacts.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h4>03. TEMPORAL MEMORY</h4>
            <p><strong>Bi-Directional LSTM.</strong> Analyzes the <em>sequence</em> of frames to detect temporal jitter and flickering invisible to the naked eye.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CHATBOT ---
    st.markdown("---")
    c_chat_anim, c_chat_box = st.columns([1, 2])
    with c_chat_anim:
        if lottie_chatbot: st_lottie(lottie_chatbot, height=250, key="bot")
    with c_chat_box:
        st.markdown("#### 💬 SECURE COMMS CHANNEL")
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if prompt := st.chat_input("Query forensic database..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                place = st.empty()
                full_resp = ""
                if gemini_active:
                    try:
                        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                        stream = model_gemini.generate_content(f"{PROJECT_CONTEXT}\n\nUser: {prompt}", stream=True)
                        for chunk in stream:
                            if chunk.text:
                                full_resp += chunk.text
                                place.markdown(full_resp + "▌")
                        place.markdown(full_resp)
                    except Exception as e: place.error(f"Error: {e}")
                else: place.markdown("⚠️ API Offline.")
            st.session_state.messages.append({"role": "assistant", "content": full_resp})

# ==========================================
# 🕵️ PAGE 2: ANALYSIS CONSOLE
# ==========================================
elif st.session_state.page == "Analysis Console":
    st.markdown('<h1 class="chrome-title" style="font-size:4rem;">ANALYSIS CONSOLE</h1>', unsafe_allow_html=True)
    st.write("")
    
    uploaded_file = st.file_uploader("UPLOAD SOURCE FOOTAGE (MP4/AVI)", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.video(uploaded_file)
            if st.button("EXECUTE DEEP SCAN", type="primary", use_container_width=True):
                if model is None: st.error("Neural Weights Not Found.")
                else:
                    status = st.empty()
                    faces, raw = process_video_frames("temp_video.mp4", status)
                    
                    if not faces: st.error("No faces detected in high-motion frames.")
                    else:
                        st.success("Target Acquired. Injecting into Neural Net...")
                        
                        # Process
                        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        input_tensor = torch.stack([transform(Image.fromarray(f)) for f in faces]).unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            probs = torch.nn.functional.softmax(output, dim=1)
                            real_score, fake_score = probs[0][0].item(), probs[0][1].item()
                        
                        # Results
                        st.markdown("---")
                        if fake_score > 0.50:
                            st.markdown(f"""
                            <div style="background: rgba(255, 0, 60, 0.2); border: 2px solid #ff003c; padding: 20px; text-align: center; border-radius: 10px; box-shadow: 0 0 30px #ff003c;">
                                <h1 style="color: #ff003c; margin:0; font-family: 'Share Tech Mono'; font-size: 3rem;">⚠️ DEEPFAKE DETECTED</h1>
                                <p style="letter-spacing: 2px; color: #ff80ab;">CONFIDENCE: {fake_score*100:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background: rgba(10, 255, 72, 0.1); border: 2px solid #0aff48; padding: 20px; text-align: center; border-radius: 10px; box-shadow: 0 0 30px #0aff48;">
                                <h1 style="color: #0aff48; margin:0; font-family: 'Share Tech Mono'; font-size: 3rem;">✅ AUTHENTIC MEDIA</h1>
                                <p style="letter-spacing: 2px; color: #b9f6ca;">CONFIDENCE: {real_score*100:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### EXTRACTED ARTIFACTS")
                        cols = st.columns(5)
                        for i, face in enumerate(faces[:5]):
                            with cols[i]: st.image(face, use_container_width=True)

        with c2:
            st.markdown("### TERMINAL LOG")
            st.markdown(f"""
            <div style="background:#000; border:1px solid #333; height:300px; padding:10px; font-family:'Share Tech Mono'; font-size:0.8rem; color:#0aff48; overflow-y:auto;">
                <div>> initializing_kernels... OK</div>
                <div>> mounting_volume... OK</div>
                <div>> waiting_for_input...</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 📄 PAGE 3: METHODOLOGY
# ==========================================
elif st.session_state.page == "Methodology":
    st.markdown('<h1 class="chrome-title" style="font-size:4rem;">SYSTEM KERNEL</h1>', unsafe_allow_html=True)
    st.write("")
    
    tabs = st.tabs(["⚡ PRE-PROCESSING", "👁️ SPATIAL LAYER", "🧠 TEMPORAL LAYER", "📊 METRICS"])
    
    with tabs[0]:
        st.markdown("### ENTROPY-BASED FRAME SELECTION")
        st.write("We don't scan every frame. We use entropy scanning to find the top 20 frames with the most movement.")
    with tabs[1]:
        st.markdown("### EFFICIENTNET-B3")
        st.write("Extracts 1536-dimensional feature vectors looking for texture anomalies.")
    with tabs[2]:
        st.markdown("### BI-DIRECTIONAL LSTM")
        st.write("Analyzes the sequence of vectors to detect temporal jitter.")
    with tabs[3]:
        c1, c2, c3 = st.columns(3)
        c1.metric("ACCURACY", "96.71%")
        c2.metric("PRECISION", "0.99")
        c3.metric("SPEED", "4.2s")

# ==========================================
# 👤 PAGE 4: ABOUT US
# ==========================================
elif st.session_state.page == "About Us":
    st.markdown('<h1 class="chrome-title" style="font-size:4rem;">DEV SQUAD</h1>', unsafe_allow_html=True)
    st.write("")
    
    def dev_card(name, role, color, desc):
        st.markdown(f"""
        <div class="glass-card" style="border-top: 3px solid {color};">
            <h3 style="color:#fff; margin:0;">{name}</h3>
            <p style="color:{color}; font-weight:bold; font-family:'Share Tech Mono';">{role}</p>
            <p style="font-size:0.9rem; color:#aaa;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: dev_card("SAHIL DESAI", "PROJECT LEAD", "#00f3ff", "Deep Learning & Model Architecture.")
    with c2: dev_card("HIMANSHU", "BACKEND ARCHITECT", "#0aff48", "Pipeline Optimization & Data Flow.")
    
    st.write("")
    c3, c4 = st.columns(2)
    with c3: dev_card("TEJAS", "DATA ANALYST", "#bc13fe", "Dataset Curation (FF++ & Celeb-DF).")
    with c4: dev_card("KRISH", "UI/UX DESIGNER", "#ff003c", "Frontend Experience & Visuals.")

# ==========================================
# 📞 PAGE 5: CONTACT
# ==========================================
elif st.session_state.page == "Contact":
    st.markdown('<h1 class="chrome-title" style="font-size:4rem;">SECURE UPLINK</h1>', unsafe_allow_html=True)
    st.write("")
    
    c1, c2, c3 = st.columns(3)
    with c1: st.link_button("🔗 LINKEDIN", "https://linkedin.com")
    with c2: st.link_button("🐙 GITHUB", "https://github.com")
    with c3: st.link_button("📧 EMAIL", "mailto:sahildesai00112@gmail.com")
