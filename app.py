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
import pandas as pd # Added for the live graph
from streamlit_lottie import st_lottie
from facenet_pytorch import MTCNN
import google.generativeai as genai

# ==========================================
# 🎨 1. PAGE CONFIG (ULTIMATE CYBERPUNK MODE)
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
GOOGLE_API_KEY = "abcd" 

gemini_active = False
try:
    if GOOGLE_API_KEY != "PASTE_YOUR_KEY_HERE":
        clean_key = GOOGLE_API_KEY.strip()
        genai.configure(api_key=clean_key)
        gemini_active = True
except Exception as e:
    st.error(f"API Setup Error: {e}")

PROJECT_CONTEXT = """
You are the AIthentic Forensic Assistant.
GOAL: Detect Deepfake videos using Temporal Inconsistency Analysis.
ARCH: EfficientNet-B3 (Spatial) + Bi-Directional LSTM (Temporal).
"""

# ==========================================
# 📂 3. ASSET LOADER & BACKGROUND HELPER
# ==========================================
def load_lottie_local(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Load Assets
lottie_left_scan = load_lottie_local("assets/Face Recognition2.json")
lottie_right_scan = load_lottie_local("assets/face recognition.json")
lottie_chatbot = load_lottie_local("assets/Live chatbot.json")
lottie_side = load_lottie_local("assets/face_loading.json")
bg_image_base64 = get_base64_of_bin_file("assets/bcimg.jpg")

# ==========================================
# 🖌️ 4. EXTREME CSS INJECTION
# ==========================================

# Determine background: Use local image if found, else fallback to grid
if bg_image_base64:
    background_style = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8)), url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    background_style = """
    [data-testid="stAppViewContainer"] {
        background-color: #050505;
        background-image: 
            linear-gradient(rgba(0, 243, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
    }
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700;900&family=Share+Tech+Mono&display=swap');

    /* --- GLOBAL VARS --- */
    :root {{
        --neon-cyan: #00f3ff;
        --neon-green: #0aff48;
        --neon-pink: #ff00ff;
        --neon-blue: #0066ff;
        --glass-bg: rgba(10, 20, 30, 0.85);
    }}

    html, body, [class*="css"] {{
        font-family: 'Rajdhani', sans-serif;
        background-color: transparent;
        color: #e0fbfc;
    }}

    /* --- BACKGROUND INJECTION --- */
    {background_style}

    /* CRT SCANLINE OVERLAY */
    [data-testid="stAppViewContainer"]::before {{
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%);
        background-size: 100% 4px;
        z-index: 2;
        pointer-events: none;
    }}

    /* --- SUPERCHARGED NEON TITLE --- */
    .mega-neon {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 900;
        font-size: 5.5rem;
        text-align: center;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 10px;
        margin-bottom: 0px;
        text-shadow: 
            0 0 5px #fff,
            0 0 10px #fff,
            0 0 20px var(--neon-cyan),
            0 0 40px var(--neon-cyan),
            0 0 80px var(--neon-cyan),
            0 0 90px var(--neon-cyan),
            0 0 100px var(--neon-cyan),
            0 0 150px var(--neon-cyan);
        animation: neon-flicker 2s infinite alternate;
    }}
    
    .neon-subtitle {{
        color: var(--neon-pink);
        text-align: center;
        font-family: 'Share Tech Mono';
        letter-spacing: 6px;
        font-size: 1.2rem;
        margin-top: -20px;
        text-shadow: 0 0 10px var(--neon-pink);
    }}

    @keyframes neon-flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{ opacity: 1; }}
        20%, 24%, 55% {{ opacity: 0.8; }}
    }}

    /* --- 3D HOVER CARDS (ARCHITECTURE) --- */
    .cyber-card-container {{
        perspective: 1000px;
        margin-bottom: 20px;
    }}
    
    .cyber-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-left: 5px solid var(--neon-cyan);
        padding: 30px;
        border-radius: 8px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    
    .cyber-card:hover {{
        transform: rotateX(10deg) rotateY(5deg) scale(1.05) translateY(-10px);
        box-shadow: 0 20px 50px rgba(0, 243, 255, 0.3), inset 0 0 20px rgba(0, 243, 255, 0.2);
        border-color: #fff;
        background: rgba(0, 243, 255, 0.1);
        z-index: 10;
        cursor: crosshair;
    }}
    
    .cyber-card h4 {{
        color: var(--neon-cyan);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: color 0.3s;
    }}
    
    .cyber-card:hover h4 {{
        color: #fff;
        text-shadow: 0 0 15px var(--neon-cyan);
    }}

    /* --- FAQ SECTION: HOLOGRAPHIC ACCORDION --- */
    .stExpander {{ background: transparent !important; border: none !important; }}
    
    .stExpander > details > summary {{
        background-color: rgba(0, 30, 50, 0.6) !important;
        color: var(--neon-cyan) !important;
        border: 1px solid var(--neon-cyan) !important;
        border-radius: 4px;
        padding: 15px !important;
        font-family: 'Share Tech Mono', monospace;
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }}
    
    .stExpander > details > summary:hover {{
        background-color: var(--neon-cyan) !important;
        color: #000 !important;
        box-shadow: 0 0 20px var(--neon-cyan);
    }}
    
    .stExpander > details > div {{
        background-color: rgba(0, 10, 15, 0.9) !important;
        border-left: 2px solid var(--neon-green) !important;
        border-right: 2px solid var(--neon-green) !important;
        color: #ddd !important;
        padding: 20px;
        margin-top: -10px;
        margin-bottom: 20px;
        box-shadow: inset 0 0 20px rgba(0, 255, 72, 0.1);
    }}

    /* --- BUTTONS --- */
    .stButton button {{
        background: transparent !important;
        border: 2px solid var(--neon-cyan) !important;
        color: var(--neon-cyan) !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        transition: 0.3s;
        border-radius: 0px !important;
        padding: 25px 0 !important;
    }}
    
    .stButton button:hover {{
        background: var(--neon-cyan) !important;
        color: #000 !important;
        box-shadow: 0 0 40px var(--neon-cyan), inset 0 0 10px #fff;
        text-shadow: 0 0 5px #fff;
        transform: scale(1.02);
    }}
    
    /* --- HUD CONTAINERS --- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(5, 10, 15, 0.7) !important;
        border: 1px solid rgba(0, 243, 255, 0.15) !important;
        box-shadow: 0 0 20px rgba(0,0,0,0.8);
        border-radius: 5px;
    }}

    /* --- TERMINAL STYLE --- */
    .terminal-box {{
        font-family: 'Share Tech Mono', monospace;
        color: var(--neon-green);
        background: #020202;
        border: 1px solid #111;
        padding: 15px;
        height: 300px;
        overflow-y: auto;
        font-size: 0.85rem;
        box-shadow: inset 0 0 20px rgba(0, 255, 72, 0.05);
    }}
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
            self.lstm = nn.LSTM(input_size=1536, hidden_size=512,
                                num_layers=1, batch_first=True, bidirectional=True)
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
    except Exception:
        return None

model = load_model()
mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=False)

# ==========================================
# 📽️ 6. VIDEO PROCESSOR
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
# 🧭 7. SIDEBAR (CYBER OPS)
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("### 🧬 OPS CENTER")
    st.markdown("---")
    
    selected_page = st.radio(
        "MODULE SELECT",
        ["Dashboard", "Analysis Console"],
        index=0 if st.session_state.page == "Dashboard" else 1,
        label_visibility="collapsed"
    )
    
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    st.markdown("---")
    st.markdown("### 📡 TELEMETRY")
    
    # SYSTEM BOOT SEQUENCE SIMULATION
    boot_text = st.empty()
    boot_logs = [
        "Initializing Kernels...",
        "Loading CUDA Drivers...",
        "Mounting Neural Net...",
        "Secure Link Established.",
        "System Ready."
    ]
    if "booted" not in st.session_state:
        for log in boot_logs:
            boot_text.markdown(f"`{log}`")
            time.sleep(0.1)
        st.session_state.booted = True
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="GPU", value=f"{random.randint(20, 45)}%")
    with c2:
        st.metric(label="RAM", value=f"{random.randint(40, 65)}%")
    
    st.markdown(f"**NET_LATENCY:** `{random.randint(12, 30)}ms`")
    
    if gemini_active:
        st.markdown("<div style='color:#0aff48; font-weight:bold; border:1px solid #0aff48; padding:5px; text-align:center;'>API STATUS: ONLINE</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#ff003c; font-weight:bold; border:1px solid #ff003c; padding:5px; text-align:center;'>API STATUS: OFFLINE</div>", unsafe_allow_html=True)

# ==========================================
# 🏠 PAGE 1: DASHBOARD
# ==========================================
if st.session_state.page == "Dashboard":
    
    # --- ULTRA TITLE ---
    st.markdown('<h1 class="mega-neon">AI THENTIC</h1>', unsafe_allow_html=True)
    st.markdown('<p class="neon-subtitle">NEURAL FORENSIC SUITE v2.0</p>', unsafe_allow_html=True)
    st.write("") # Spacer

    # --- TOP ROW: ANIMATIONS ---
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if lottie_left_scan: st_lottie(lottie_left_scan, height=180, key="l1")
        with col2:
            st.markdown("""
                <div style="text-align: center; padding-top: 20px;">
                    <h3 style="color: #fff; text-shadow: 0 0 10px #00f3ff;">DIGITAL INTEGRITY VERIFICATION</h3>
                    <p style="color: #aaa; font-family: 'Share Tech Mono';">
                        DEPLOYING BI-DIRECTIONAL LSTM ARRAYS FOR DEEPFAKE ARTIFACT DETECTION.
                        ANALYZING SPATIAL INCONSISTENCIES AND TEMPORAL JITTER.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            if st.button(">> INITIALIZE ANALYSIS MODULE <<", type="primary", use_container_width=True):
                st.session_state.page = "Analysis Console"
                st.rerun()
        with col3:
             if lottie_right_scan: st_lottie(lottie_right_scan, height=180, key="r1")

    # --- ARCHITECTURE (3D CARDS) ---
    st.markdown("<h3 style='text-align:center; color: var(--neon-cyan); margin-top: 50px; letter-spacing:4px;'>SYSTEM ARCHITECTURE</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666; font-size:0.8rem; margin-bottom:30px;'>HOVER CARDS FOR DIAGNOSTICS</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
            <div class="cyber-card-container">
                <div class="cyber-card">
                    <h4>01. ACTIVE SAMPLING</h4>
                    <p>High-Entropy Frame Extraction. Algorithm actively discards static data to focus solely on high-motion vectors where artifacts occur.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
            <div class="cyber-card-container">
                <div class="cyber-card">
                    <h4>02. TEMPORAL MEMORY</h4>
                    <p>Bi-Directional LSTM Core. Analyzes frame-to-frame inconsistencies in the time domain to detect temporal jitter.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
            <div class="cyber-card-container">
                <div class="cyber-card">
                    <h4>03. SPATIAL SCAN</h4>
                    <p>EfficientNet-B3 Backbone. Detects blending boundaries, resolution mismatches, and warping artifacts.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- FAQ SECTION ---
    st.markdown("---")
    st.markdown("<h3 style='text-align:center; color: var(--neon-cyan); letter-spacing:4px;'>FORENSIC KNOWLEDGE BASE</h3>", unsafe_allow_html=True)
    
    col_faq_L, col_faq_R = st.columns([1, 1])
    
    with col_faq_L:
        with st.expander("❓ What exactly is a Deepfake?"):
            st.write("""Deepfakes are synthetic media generated by AI, specifically **Generative Adversarial Networks (GANs)**. They use two neural networks competing against each other: a generator and a discriminator. 

[Image of GAN Architecture]
""")
        with st.expander("⚙️ How are Deepfakes generated?"):
            st.write("""Most are created using an **Autoencoder** architecture which compresses the input face into a latent space and reconstructs it as the target face. """)
        with st.expander("🤔 Difference between DeepFace and Deepfakes?"):
            st.write("""**DeepFace** is a facial *recognition* system (Identity Verification). **Deepfakes** are synthetic media (Identity Manipulation).""")
            
    with col_faq_R:
        with st.expander("🕵️ What are 'Deepfake Artifacts'?"):
            st.write("""Artifacts are the 'glitches' AI leaves behind, such as: **Blending Boundaries** (where the fake face meets the real head) and **Temporal Jitter** (flickering between frames). """)
        with st.expander("👁️ Why is 'Active Sampling' important?"):
            st.write("""Deepfakes often look perfect in static frames. Active Sampling ignores static frames and forces the model to analyze only high-movement frames where the AI is most likely to fail.""")

    # --- CHATBOT SECTION ---
    st.markdown("---")
    c_chat_anim, c_chat_box = st.columns([1, 2])
    
    with c_chat_anim:
        st.markdown("<h4 style='color: var(--neon-cyan); text-align: center;'>AI ASSISTANT LINK</h4>", unsafe_allow_html=True)
        if lottie_chatbot: st_lottie(lottie_chatbot, height=250, key="bot")
        else: st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDdtY255d3F6Y255d3F6/giphy.gif", use_container_width=True)

    with c_chat_box:
        with st.container(border=True):
            st.markdown("**SECURE COMMS CHANNEL**")
            if "messages" not in st.session_state: st.session_state.messages = []
            
            for msg in st.session_state.messages:
                role_color = "#0aff48" if msg["role"] == "assistant" else "#00f3ff"
                st.markdown(f"<span style='color:{role_color}; font-weight:bold;'>{msg['role'].upper()}:</span> {msg['content']}", unsafe_allow_html=True)
            
            prompt = st.chat_input("Query forensic database...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                response_text = "ACCESS DENIED. API KEY INVALID."
                if gemini_active:
                    try:
                        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                        full_prompt = f"{PROJECT_CONTEXT}\n\nUser Question: {st.session_state.messages[-1]['content']}"
                        response = model_gemini.generate_content(full_prompt)
                        response_text = response.text
                    except Exception as e:
                        try:
                            model_gemini = genai.GenerativeModel("gemini-pro")
                            response = model_gemini.generate_content(full_prompt)
                            response_text = response.text
                        except Exception as e2:
                            response_text = f"CONNECTION ERROR: Please run 'pip install -U google-generativeai' in terminal. Error details: {e2}"
                else:
                    time.sleep(1)
                    response_text = f"Simulating analysis... [API Key required]"
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()

# ==========================================
# 🕵️ PAGE 2: ANALYSIS CONSOLE
# ==========================================
elif st.session_state.page == "Analysis Console":
    
    st.markdown('<h1 class="mega-neon" style="font-size:3rem;">ANALYSIS CONSOLE</h1>', unsafe_allow_html=True)
    st.write("")
    
    uploaded_file = st.file_uploader(" ", type=["mp4", "avi", "mov"])
    st.markdown("<p style='text-align:center; color:#666; font-size: 0.8rem;'>SUPPORTED FORMATS: MP4, AVI, MOV // MAX SIZE: 200MB</p>", unsafe_allow_html=True)

    if uploaded_file:
        with open("temp_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        
        col_vid, col_data = st.columns([1.5, 1])
        
        with col_vid:
            with st.container(border=True):
                st.markdown("<div style='position:absolute; top:10px; left:10px; color:red; font-size:0.7rem;'>REC ●</div>", unsafe_allow_html=True)
                st.video(uploaded_file)
                analyze_btn = st.button("INITIATE DEEP SCAN", type="primary", use_container_width=True)

        with col_data:
            with st.container(border=True):
                st.markdown("**TERMINAL LOG**")
                terminal_placeholder = st.empty()
                terminal_placeholder.markdown('<div class="terminal-box">_WAITING FOR INPUT...</div>', unsafe_allow_html=True)

        if analyze_btn:
            if model is None:
                terminal_placeholder.markdown('<div class="terminal-box" style="color:red;">[FATAL] NEURAL WEIGHTS NOT FOUND.<br>Please ensure .pth file is in directory.</div>', unsafe_allow_html=True)
            else:
                status_box = st.empty()
                faces, raw = process_video_frames("temp_video.mp4", status_box)
                
                if not faces:
                    terminal_placeholder.markdown('<div class="terminal-box" style="color:red;">[ERROR] NO FACIAL DATA EXTRACTED.<br>Video may be too short or face obscured.</div>', unsafe_allow_html=True)
                else:
                    terminal_placeholder.markdown('<div class="terminal-box">[INFO] FACES EXTRACTED.<br>[INFO] INJECTING INTO NEURAL NET...</div>', unsafe_allow_html=True)
                    
                    # === ADD-ON: LIVE NEURAL TELEMETRY GRAPH ===
                    st.write("")
                    st.markdown("**⚡ LIVE NEURAL TELEMETRY**")
                    graph_place = st.empty()
                    
                    # Simulate "Reading" frames before final result
                    chart_data = pd.DataFrame(columns=["Integrity"])
                    for x in range(30):
                        new_row = pd.DataFrame({"Integrity": [random.uniform(0.4, 0.9)]})
                        chart_data = pd.concat([chart_data, new_row], ignore_index=True)
                        # Create a cool area chart
                        graph_place.area_chart(chart_data, color="#00f3ff", height=150)
                        time.sleep(0.05)
                    # ===========================================

                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    input_tensor = torch.stack([transform(Image.fromarray(f)) for f in faces]).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        real_score, fake_score = probs[0][0].item(), probs[0][1].item()

                    st.write("---")
                    
                    if fake_score > 0.50:
                        st.markdown(f"""
                        <div style="background: rgba(255, 0, 60, 0.2); border: 2px solid #ff003c; padding: 20px; text-align: center; border-radius: 10px; box-shadow: 0 0 30px #ff003c;">
                            <h1 style="color: #ff003c; margin:0; font-family: 'Share Tech Mono'; font-size: 3rem; text-shadow: 0 0 20px #ff003c;">⚠️ DEEPFAKE DETECTED</h1>
                            <p style="letter-spacing: 2px; color: #ff80ab;">CONFIDENCE: {fake_score*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.toast("⚠️ THREAT DETECTED: DEEPFAKE SIGNATURE FOUND", icon="🚨")
                    else:
                        st.markdown(f"""
                        <div style="background: rgba(10, 255, 72, 0.1); border: 2px solid #0aff48; padding: 20px; text-align: center; border-radius: 10px; box-shadow: 0 0 30px #0aff48;">
                            <h1 style="color: #0aff48; margin:0; font-family: 'Share Tech Mono'; font-size: 3rem; text-shadow: 0 0 20px #0aff48;">✅ AUTHENTIC MEDIA</h1>
                            <p style="letter-spacing: 2px; color: #b9f6ca;">CONFIDENCE: {real_score*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.toast("✅ SYSTEM SECURE: MEDIA VERIFIED", icon="🛡️")

                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"**REAL PROBABILITY**")
                        st.progress(real_score)
                    with m2:
                        st.markdown(f"**FAKE PROBABILITY**")
                        st.progress(fake_score)
                    
                    terminal_placeholder.markdown('<div class="terminal-box" style="color: var(--neon-cyan);">[SUCCESS] ANALYSIS COMPLETE.<br>[LOG] REPORT GENERATED.<br>[LOG] ARTIFACTS ARCHIVED.</div>', unsafe_allow_html=True)
                    
                    st.write("---")
                    st.subheader("Extracted Artifacts")
                    
                    st.markdown("""<style>.stImage { border: 1px solid var(--neon-cyan); transition: transform 0.3s; } .stImage:hover { transform: scale(1.1); box-shadow: 0 0 15px var(--neon-cyan); }</style>""", unsafe_allow_html=True)
                    
                    f_cols = st.columns(10)
                    for i, face in enumerate(faces[:10]):
                        with f_cols[i]: st.image(face, use_container_width=True)
