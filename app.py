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
import gdown  # Standard library for Drive downloads

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

# Load Assets
lottie_left_scan = load_lottie_local("assets/animation1.json")
lottie_right_scan = load_lottie_local("assets/animation2.json")
lottie_chatbot = load_lottie_local("assets/animation3.json")
bg_image_base64 = get_base64_of_bin_file("assets/back_ground_img.jpg")

# ==========================================
# 🖌️ 4. ULTRA-MODERN CSS
# ==========================================

# Determine background
if bg_image_base64:
    background_style = f"""
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.9)), url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    background_style = """
    [data-testid="stAppViewContainer"] {
        background-color: #050505;
        background-image: linear-gradient(rgba(0, 243, 255, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 243, 255, 0.05) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;800&family=Share+Tech+Mono&display=swap');

    :root {{ --neon-blue: #00f3ff; --neon-purple: #bc13fe; --neon-green: #0aff48; --neon-red: #ff003c; }}

    html, body, [class*="css"] {{ font-family: 'Rajdhani', sans-serif; color: #e0fbfc; font-size: 16px; }}

    /* --- BACKGROUND --- */
    {background_style}

    /* --- NEON TITLE EFFECT --- */
    .neon-title {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 800;
        font-size: 4.5rem;
        text-align: center;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 8px;
        margin-bottom: 5px;
        text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px var(--neon-blue), 0 0 40px var(--neon-blue);
        animation: flicker 3s infinite alternate;
    }}

    @keyframes flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{ opacity: 1; }}
        20%, 24%, 55% {{ opacity: 0.4; }}
    }}

    /* --- SUBTITLE --- */
    .tech-subtitle {{
        font-family: 'Share Tech Mono', monospace;
        color: var(--neon-purple);
        text-align: center;
        font-size: 1.1rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: -10px;
        opacity: 0.9;
        text-shadow: 0 0 10px rgba(188, 19, 254, 0.5);
    }}

    /* --- INTEGRITY BOX --- */
    .integrity-box {{
        text-align: center; 
        padding: 20px;
        border: 1px solid rgba(0, 243, 255, 0.3);
        background: rgba(0, 20, 30, 0.7);
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);
    }}
    .integrity-box h3 {{ font-size: 1.4rem; color: #fff; margin-bottom: 5px; }}
    .integrity-box p {{ font-size: 0.85rem; color: #aaa; font-family: 'Share Tech Mono'; }}

    /* --- CARDS --- */
    .cyber-card {{
        background: rgba(10, 15, 20, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid var(--neon-blue);
        padding: 20px; 
        border-radius: 6px; 
        transition: transform 0.3s;
        height: 100%;
    }}
    .cyber-card:hover {{
        transform: translateY(-5px);
        background: rgba(0, 243, 255, 0.05);
        box-shadow: 0 10px 20px rgba(0, 243, 255, 0.1);
    }}
    .cyber-card h4 {{ color: var(--neon-blue); font-size: 1.2rem; font-weight: 700; margin-bottom: 10px; }}
    .cyber-card p {{ color: #ccc; font-size: 0.85rem; line-height: 1.5; }}

    /* --- FAQ EXPANDER STYLING --- */
    .stExpander {{ background: transparent !important; border: none !important; }}
    .stExpander > details > summary {{
        background-color: rgba(0, 30, 50, 0.6) !important;
        color: var(--neon-blue) !important;
        border: 1px solid var(--neon-blue) !important;
        border-radius: 4px;
        padding: 10px 15px !important;
        font-family: 'Share Tech Mono', monospace;
        transition: all 0.3s ease;
    }}
    .stExpander > details > summary:hover {{
        background-color: var(--neon-blue) !important;
        color: #000 !important;
    }}
    .stExpander > details > div {{
        background-color: rgba(0, 10, 15, 0.9) !important;
        color: #ddd !important;
        border-left: 2px solid var(--neon-green) !important;
        padding: 15px;
    }}

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] .stRadio label {{ padding: 8px 0; }}
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
        font-size: 1.0rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover p {{
        color: #fff;
        padding-left: 8px;
        text-shadow: 0 0 10px var(--neon-blue);
    }}

    /* --- BUTTONS --- */
    .stButton button {{
        background: transparent !important;
        border: 1px solid var(--neon-blue) !important;
        color: var(--neon-blue) !important;
        font-family: 'Share Tech Mono' !important;
        font-size: 1rem !important;
        padding: 10px 20px !important;
        transition: all 0.3s;
    }}
    .stButton button:hover {{
        background: rgba(0, 243, 255, 0.1) !important;
        box-shadow: 0 0 20px var(--neon-blue);
    }}
    
    /* --- TERMINAL --- */
    .terminal-box {{
        background: #050505;
        border: 1px solid #333;
        padding: 15px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        color: var(--neon-green);
        border-left: 3px solid var(--neon-green);
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
            print(f"Downloading model from Drive...")
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    try:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return model
        return None
    except Exception as e:
        st.error(f"Error loading weights: {e}")
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
# 🧭 7. SIDEBAR
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; border-bottom: 1px solid #333; padding-bottom: 20px; margin-bottom: 20px;">
        <h2 style="color: #fff; margin:0; letter-spacing: 2px; font-size: 1.6rem; text-shadow: 0 0 10px #00f3ff;">OPS CENTER</h2>
        <p style="color: #00f3ff; margin:0; font-size: 0.75rem; letter-spacing: 2px; font-weight: bold;">V.2.0.4</p>
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
    
    # TELEMETRY
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="color: #00f3ff; font-size: 1.2rem; margin-right: 10px;">📡</span>
        <span style="color: #fff; font-family: 'Share Tech Mono'; letter-spacing: 1px; font-size: 1.0rem; font-weight: 700;">LIVE TELEMETRY</span>
    </div>
    """, unsafe_allow_html=True)

    gpu_load = random.randint(30, 85)
    ram_load = random.randint(40, 65)
    temp = random.randint(45, 75)

    st.markdown(f"""
    <style>
        .meter-label {{ display: flex; justify-content: space-between; color: #fff; font-size: 0.8rem; margin-bottom: 2px; }}
        .meter-bar {{ width: 100%; height: 6px; background: #222; border-radius: 3px; margin-bottom: 10px; }}
        .meter-fill {{ height: 100%; border-radius: 3px; box-shadow: 0 0 5px currentColor; }}
    </style>
    <div>
        <div class="meter-label"><span>GPU</span><span style="color:#00f3ff">{gpu_load}%</span></div>
        <div class="meter-bar"><div class="meter-fill" style="width:{gpu_load}%; background:#00f3ff; color:#00f3ff"></div></div>
    </div>
    <div>
        <div class="meter-label"><span>RAM</span><span style="color:#bc13fe">{ram_load}%</span></div>
        <div class="meter-bar"><div class="meter-fill" style="width:{ram_load}%; background:#bc13fe; color:#bc13fe"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.8rem; color: #fff; margin-bottom: 5px; font-family: Rajdhani;'>NEURAL_FLUX</p>", unsafe_allow_html=True)
    st.line_chart(pd.DataFrame(np.random.randn(20, 1), columns=['a']), height=80, color="#0aff48") 

    # API BADGE
    st.write("")
    if gemini_active:
        st.markdown('<div style="background: rgba(10,255,72,0.1); border: 1px solid #0aff48; text-align: center; padding: 8px; border-radius: 4px;"><span style="color: #0aff48; font-weight: bold; font-size: 0.8rem;">SYSTEM ONLINE</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: rgba(255,0,60,0.1); border: 1px solid #ff003c; text-align: center; padding: 8px; border-radius: 4px;"><span style="color: #ff003c; font-weight: bold; font-size: 0.8rem;">OFFLINE MODE</span></div>', unsafe_allow_html=True)

# ==========================================
# 🏠 PAGE 1: DASHBOARD
# ==========================================
if st.session_state.page == "Dashboard":
    
    st.markdown('<h1 class="neon-title">AI THENTIC</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tech-subtitle">>> NEURAL FORENSIC SUITE v2.0 <<</p>', unsafe_allow_html=True)
    st.write("") 

    # --- TOP ROW ---
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if lottie_left_scan: st_lottie(lottie_left_scan, height=150, key="l1")
        with col2:
            st.markdown("""
                <div class="integrity-box">
                    <h3 style="color: #fff; text-shadow: 0 0 10px #00f3ff;">DIGITAL INTEGRITY VERIFICATION</h3>
                    <p>DEPLOYING BI-DIRECTIONAL LSTM ARRAYS FOR DEEPFAKE ARTIFACT DETECTION.</p>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button(">> INITIALIZE ANALYSIS MODULE <<", type="primary", use_container_width=True):
                st.session_state.page = "Analysis Console"
                st.rerun()
        with col3:
             if lottie_right_scan: st_lottie(lottie_right_scan, height=150, key="r1")

    # --- ARCHITECTURE ---
    st.markdown("<h3 style='text-align:center; color: var(--neon-blue); margin-top: 30px; font-size: 1.5rem; letter-spacing: 2px;'>SYSTEM ARCHITECTURE</h3>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("""
        <div class="cyber-card">
            <h4>01. ACTIVE SAMPLING</h4>
            <p>High-Entropy Frame Extraction. Algorithm discards static data to focus solely on high-motion vectors.</p>
        </div>
    """, unsafe_allow_html=True)
    
    with c2: st.markdown("""
        <div class="cyber-card">
            <h4>02. TEMPORAL MEMORY</h4>
            <p>Bi-Directional LSTM Core. Analyzes frame-to-frame inconsistencies in the time domain to detect jitter.</p>
        </div>
    """, unsafe_allow_html=True)
    
    with c3: st.markdown("""
        <div class="cyber-card">
            <h4>03. SPATIAL SCAN</h4>
            <p>EfficientNet-B3 Backbone. Detects blending boundaries, resolution mismatches, and warping artifacts.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- FAQ (RESTORED) ---
    st.markdown("---")
    st.markdown("<h3 style='text-align:center; color: var(--neon-blue); font-size: 1.5rem; letter-spacing: 2px;'>FORENSIC KNOWLEDGE BASE</h3>", unsafe_allow_html=True)
    
    col_faq_L, col_faq_R = st.columns([1, 1])
    with col_faq_L:
        with st.expander("❓ What exactly is a Deepfake?"): st.write("""Deepfakes are synthetic media generated by AI, specifically **Generative Adversarial Networks (GANs)**.""")
        with st.expander("⚙️ How are Deepfakes generated?"): st.write("""Most are created using an **Autoencoder** architecture which compresses the input face into a latent space and reconstructs it as the target face. """)
        with st.expander("🤔 Difference between DeepFace and Deepfakes?"): st.write("""**DeepFace** is a facial *recognition* system. **Deepfakes** are synthetic media.""")
    with col_faq_R:
        with st.expander("🕵️ What are 'Deepfake Artifacts'?"): st.write("""Artifacts are the 'glitches' AI leaves behind, such as: **Blending Boundaries** and **Temporal Jitter**.""")
        with st.expander("👁️ Why is 'Active Sampling' important?"): st.write("""Active Sampling ignores static frames and forces the model to analyze only high-movement frames where the AI is most likely to fail.""")

    # --- CHATBOT SECTION ---
    st.markdown("---")
    c_chat_anim, c_chat_box = st.columns([1, 2])
    
    with c_chat_anim:
        if lottie_chatbot: st_lottie(lottie_chatbot, height=200, key="bot")
        else: st.markdown("⚠️ Animation Assets Missing")

    with c_chat_box:
        with st.container(border=True):
            st.markdown("**SECURE COMMS CHANNEL**")
            
            if "messages" not in st.session_state: st.session_state.messages = []
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
            if prompt := st.chat_input("Query forensic database..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    if gemini_active:
                        try:
                            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                            full_prompt = f"{PROJECT_CONTEXT}\n\nUser Question: {prompt}"
                            response_stream = model_gemini.generate_content(full_prompt, stream=True)
                            for chunk in response_stream:
                                if chunk.text:
                                    full_response += chunk.text
                                    message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                        except Exception as e:
                            message_placeholder.error(f"Error: {str(e)}")
                    else:
                        message_placeholder.markdown("⚠️ SYSTEM OFFLINE. API KEY REQUIRED.")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# 🕵️ PAGE 2: ANALYSIS CONSOLE
# ==========================================
elif st.session_state.page == "Analysis Console":
    
    st.markdown('<h1 class="neon-title" style="font-size:3rem;">ANALYSIS CONSOLE</h1>', unsafe_allow_html=True)
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
                terminal_placeholder.markdown('<div class="terminal-box" style="color:red;">[FATAL] NEURAL WEIGHTS NOT FOUND.</div>', unsafe_allow_html=True)
            else:
                status_box = st.empty()
                faces, raw = process_video_frames("temp_video.mp4", status_box)
                
                if not faces:
                    terminal_placeholder.markdown('<div class="terminal-box" style="color:red;">[ERROR] NO FACIAL DATA EXTRACTED.</div>', unsafe_allow_html=True)
                else:
                    terminal_placeholder.markdown('<div class="terminal-box">[INFO] FACES EXTRACTED.<br>[INFO] INJECTING INTO NEURAL NET...</div>', unsafe_allow_html=True)
                    
                    st.write("")
                    st.markdown("**⚡ LIVE NEURAL TELEMETRY**")
                    graph_place = st.empty()
                    chart_data = pd.DataFrame(columns=["Integrity"])
                    for x in range(20):
                        new_row = pd.DataFrame({"Integrity": [random.uniform(0.4, 0.9)]})
                        chart_data = pd.concat([chart_data, new_row], ignore_index=True)
                        graph_place.area_chart(chart_data, color="#00f3ff", height=100)
                        time.sleep(0.05)

                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    input_tensor = torch.stack([transform(Image.fromarray(f)) for f in faces]).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        real_score, fake_score = probs[0][0].item(), probs[0][1].item()

                    st.write("---")
                    
                    if fake_score > 0.50:
                        st.markdown(f"""<div style="background: rgba(255, 0, 60, 0.2); border: 1px solid #ff003c; padding: 15px; text-align: center; border-radius: 8px;"><h2 style="color: #ff003c; margin:0; font-family: 'Share Tech Mono';">⚠️ DEEPFAKE DETECTED</h2><p style="letter-spacing: 1px; color: #ff80ab;">CONFIDENCE: {fake_score*100:.2f}%</p></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div style="background: rgba(10, 255, 72, 0.1); border: 1px solid #0aff48; padding: 15px; text-align: center; border-radius: 8px;"><h2 style="color: #0aff48; margin:0; font-family: 'Share Tech Mono';">✅ AUTHENTIC MEDIA</h2><p style="letter-spacing: 1px; color: #b9f6ca;">CONFIDENCE: {real_score*100:.2f}%</p></div>""", unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    with m1: st.markdown(f"**REAL PROBABILITY**"); st.progress(real_score)
                    with m2: st.markdown(f"**FAKE PROBABILITY**"); st.progress(fake_score)
                    
                    terminal_placeholder.markdown('<div class="terminal-box" style="color: var(--neon-cyan);">[SUCCESS] ANALYSIS COMPLETE.</div>', unsafe_allow_html=True)
                    
                    st.write("---")
                    st.subheader("Extracted Artifacts")
                    st.markdown("""<style>.stImage { border: 1px solid var(--neon-cyan); }</style>""", unsafe_allow_html=True)
                    f_cols = st.columns(10)
                    for i, face in enumerate(faces[:10]):
                        with f_cols[i]: st.image(face, use_container_width=True)

# ==========================================
# 📄 PAGE 3: METHODOLOGY
# ==========================================
elif st.session_state.page == "Methodology":
    st.markdown('<h1 class="neon-title" style="font-size:3rem;">SYSTEM KERNEL</h1>', unsafe_allow_html=True)
    st.write("")

    # --- 1. VISUAL PIPELINE ---
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,20,30,0.6); padding: 15px; border-radius: 8px; border: 1px solid #00f3ff; margin-bottom: 30px;">
        <div style="text-align:center; opacity: 0.8;">📹<br><span style="font-size:0.7rem; color:#aaa;">INPUT</span></div>
        <div style="color: #00f3ff;">➜</div>
        <div style="text-align:center; color:#00f3ff;">⚡<br><span style="font-size:0.7rem;">SAMPLING</span></div>
        <div style="color: #00f3ff;">➜</div>
        <div style="text-align:center; color:#ff00ff;">👁️<br><span style="font-size:0.7rem;">SPATIAL</span></div>
        <div style="color: #00f3ff;">➜</div>
        <div style="text-align:center; color:#0aff48;">🧠<br><span style="font-size:0.7rem;">TEMPORAL</span></div>
        <div style="color: #00f3ff;">➜</div>
        <div style="text-align:center; color:#fff;">🛡️<br><span style="font-size:0.7rem;">VERDICT</span></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚡ PRE-PROCESSING", "👁️ SPATIAL LAYER", "🧠 TEMPORAL LAYER"])

    with tab1:
        st.markdown("### ENTROPY-BASED FRAME SELECTION")
        st.write("The system calculates pixel difference to find the Top 20 High-Motion Frames.")
    with tab2:
        st.markdown("### EFFICIENTNET-B3")
        st.write("Extracts 1536-dimensional feature vectors from each frame.")
    with tab3:
        st.markdown("### BI-DIRECTIONAL LSTM")
        st.write("Analyzes the sequence of vectors to detect temporal jitter.")

# ==========================================
# 👤 PAGE 4: ABOUT US
# ==========================================
elif st.session_state.page == "About Us":
    st.markdown('<h1 class="neon-title" style="font-size:3rem;">DEV TEAM</h1>', unsafe_allow_html=True)
    st.write("")
    
    def dev_card(name, role, color, desc):
        st.markdown(f"""
        <div class="cyber-card" style="border-top: 3px solid {color}; padding: 15px;">
            <h3 style="color:#fff; margin:0; font-size:1.2rem;">{name}</h3>
            <p style="color:{color}; font-weight:bold; font-family:'Share Tech Mono'; font-size:0.9rem;">{role}</p>
            <p style="font-size:0.8rem; color:#aaa;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: dev_card("SAHIL DESAI", "PROJECT LEAD", "#00f3ff", "Deep Learning & Model Architecture.")
    with c2: dev_card("HIMANSHU", "BACKEND ARCHITECT", "#0aff48", "Pipeline Optimization.")
    
    st.write("")
    c3, c4 = st.columns(2)
    with c3: dev_card("TEJAS", "DATA ANALYST", "#bc13fe", "Dataset Curation.")
    with c4: dev_card("KRISH", "UI/UX DESIGNER", "#ff003c", "Frontend Visuals.")

# ==========================================
# 📞 PAGE 5: CONTACT
# ==========================================
elif st.session_state.page == "Contact":
    st.markdown('<h1 class="neon-title" style="font-size:3rem;">SECURE UPLINK</h1>', unsafe_allow_html=True)
    st.write("")
    
    with st.container(border=True):
        st.markdown("### 📡 ESTABLISH CONNECTION")
        c1, c2, c3 = st.columns(3)
        with c1: st.link_button("🔗 LINKEDIN", "https://linkedin.com")
        with c2: st.link_button("🐙 GITHUB", "https://github.com")
        with c3: st.link_button("📧 EMAIL", "mailto:sahildesai00112@gmail.com")
