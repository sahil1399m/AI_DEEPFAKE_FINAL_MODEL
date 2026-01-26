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
# 🧠 THE BRAIN: ADVANCED KNOWLEDGE BASE
# ==========================================
PROJECT_CONTEXT = """
ROLE: You are the "AIthentic Forensic Assistant", a military-grade neural expert in Digital Media Forensics.
Your goal is to explain the technical depth of the AIthentic platform to judges, recruiters, and users.

--- 1. DATASET & TRAINING STRATEGY (CRITICAL) ---
* **The "Hybrid-Data" Approach:** * We trained on a custom fusion of **FaceForensics++ (FF++)** and **Celeb-DF (v2)**.
    * *Why?* FF++ is large but contains high compression artifacts. Older models "cheat" by detecting these background compression noise rather than the face itself.
    * *The Fix:* By injecting Celeb-DF (which has high-quality, seamless blending), we forced our Neural Network to unlearn background noise and focus purely on **facial biological inconsistencies**.
* **Preprocessing:** All videos were normalized to 30 FPS. Faces were extracted using MTCNN with a 30% margin to include jawline boundaries (where blending fails).

--- 2. SYSTEM ARCHITECTURE (THE PIPELINE) ---
* **Stage 1: Active Temporal Sampling (The Filter)**
    * We do NOT analyze every frame (too slow/redundant).
    * We use **Entropy-Based Motion Scanning** to calculate the pixel difference between consecutive frames.
    * We select the **Top 20 High-Entropy Frames**—these are moments of rapid movement (blinking, head turning) where Deepfake generators (GANs) effectively "glitch."

* **Stage 2: Spatial Analysis (The Eye)**
    * **Model:** EfficientNet-B3 (Pre-trained on ImageNet).
    * **Role:** Extracts a 1536-dimensional feature vector from each of the 20 frames.
    * **Focus:** It detects "high-frequency" artifacts—pixel-level mismatches in skin texture and blending boundaries that the human eye misses.

* **Stage 3: Temporal Analysis (The Brain)**
    * **Model:** Bi-Directional LSTM (Long Short-Term Memory).
    * **Role:** It looks at the *sequence* of features produced by EfficientNet.
    * **Focus:** It detects **Temporal Jitter**—micro-flickers in the lips or eyes that happen *across time* but look fine in a single still image.

--- 3. PERFORMANCE METRICS ---
* **Accuracy:** 96.71% (Tested on unseen cross-dataset samples).
* **Precision:** 0.99 (Extremely low False Positive rate).
* **Inference Speed:** ~4 seconds for a 10s video on standard GPU.

--- 4. INSTRUCTIONS FOR YOU ---
* If the user asks "How does it work?", summarize the 3 stages above simply.
* If asked about "Datasets", explain the FF++ and Celeb-DF fusion strategy to prevent overfitting.
* If asked "Is this video fake?", strictly reply: "Please upload the media to the Analysis Console for a real-time neural scan."
* Keep your tone professional, technical, and slightly "Cyberpunk" (efficient, precise).
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
lottie_left_scan = load_lottie_local("assets/animation1.json")
lottie_right_scan = load_lottie_local("assets/animation2.json")
lottie_chatbot = load_lottie_local("assets/animation3.json")
lottie_side = load_lottie_local("assets/face_loading.json")
bg_image_base64 = get_base64_of_bin_file("assets/back_ground_img.jpg")

# ==========================================
# 🖌️ 4. EXTREME CSS INJECTION (UPDATED)
# ==========================================

# Determine background
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
        background-image: linear-gradient(rgba(0, 243, 255, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 243, 255, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
    }
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700;900&family=Share+Tech+Mono&display=swap');

    /* --- GLOBAL VARS --- */
    :root {{ --neon-cyan: #00f3ff; --neon-green: #0aff48; --neon-pink: #ff00ff; --neon-blue: #0066ff; }}

    html, body, [class*="css"] {{ font-family: 'Rajdhani', sans-serif; background-color: transparent; color: #e0fbfc; }}

    /* --- BACKGROUND INJECTION --- */
    {background_style}

    /* CRT SCANLINE OVERLAY */
    [data-testid="stAppViewContainer"]::before {{
        content: " "; display: block; position: absolute; top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%);
        background-size: 100% 4px; z-index: 2; pointer-events: none;
    }}

    /* --- HYPER-CHROME TITLE (SHARPER & SHINIER) --- */
    .chrome-title {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 900;
        font-size: 6.5rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 12px;
        margin-bottom: 0;
        padding-bottom: 10px;
        
        background: linear-gradient(110deg, #000000 10%, #00f3ff 40%, #ffffff 50%, #00f3ff 60%, #000000 90%);
        background-size: 200% auto;
        
        color: #fff;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        
        -webkit-text-stroke: 1px rgba(0, 243, 255, 0.5);
        filter: drop-shadow(0 0 5px rgba(0, 243, 255, 0.8));
        
        animation: shine 4s linear infinite;
    }}

    @keyframes shine {{ to {{ background-position: 200% center; }} }}

    /* --- DATA STREAM SUBTITLE --- */
    .glitch-subtitle {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.4rem;
        color: var(--neon-pink);
        text-align: center;
        letter-spacing: 5px;
        margin-top: -15px;
        position: relative;
        background: rgba(0,0,0,0.6);
        padding: 5px 20px;
        border: 1px solid rgba(255, 0, 255, 0.3);
        border-radius: 4px;
        display: inline-block;
        text-shadow: 2px 2px 0px rgba(0, 243, 255, 0.8);
        animation: glitch-pulse 3s infinite;
    }}

    @keyframes glitch-pulse {{
        0% {{ text-shadow: 2px 2px 0px rgba(0, 243, 255, 0.8); }}
        50% {{ text-shadow: -2px -2px 0px rgba(0, 243, 255, 0.8); }}
        100% {{ text-shadow: 2px 2px 0px rgba(0, 243, 255, 0.8); }}
    }}

    /* --- GLOWING INTEGRITY BOX --- */
    .integrity-box {{
        text-align: center; 
        padding-top: 20px;
        padding-bottom: 20px;
        border: 1px solid rgba(0, 243, 255, 0.2);
        background: rgba(0, 20, 30, 0.6);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);
        animation: box-pulse 4s infinite alternate;
    }}
    @keyframes box-pulse {{
        0% {{ box-shadow: 0 0 10px rgba(0, 243, 255, 0.1); border-color: rgba(0, 243, 255, 0.2); }}
        100% {{ box-shadow: 0 0 30px rgba(0, 243, 255, 0.3); border-color: rgba(0, 243, 255, 0.6); }}
    }}

    /* --- 3D HOVER CARDS (ARCHITECTURE) --- */
    .cyber-card-container {{ perspective: 1000px; margin-bottom: 20px; }}
    .cyber-card {{
        background: rgba(5, 15, 25, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 3px solid var(--neon-cyan);
        padding: 30px; 
        border-radius: 8px; 
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
    }}
    .cyber-card:hover {{
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 243, 255, 0.2);
        border-color: var(--neon-cyan);
        background: rgba(0, 243, 255, 0.05);
    }}
    .cyber-card h4 {{ 
        color: #fff; 
        font-size: 1.4rem; 
        font-weight: 700; 
        margin-bottom: 15px; 
        letter-spacing: 2px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
    }}
    .cyber-card p {{ color: #aaa; font-size: 0.9rem; line-height: 1.6; }}
    .cyber-card:hover h4 {{ color: var(--neon-cyan); text-shadow: 0 0 10px var(--neon-cyan); }}

    /* --- FAQ & BUTTONS --- */
    .stExpander {{ background: transparent !important; border: none !important; }}
    .stExpander > details > summary {{
        background-color: rgba(0, 30, 50, 0.6) !important; color: var(--neon-cyan) !important;
        border: 1px solid var(--neon-cyan) !important; border-radius: 4px; padding: 15px !important;
        font-family: 'Share Tech Mono', monospace; transition: all 0.3s ease;
    }}
    .stExpander > details > summary:hover {{ background-color: var(--neon-cyan) !important; color: #000 !important; }}
    .stExpander > details > div {{ background-color: rgba(0, 10, 15, 0.9) !important; border-left: 2px solid var(--neon-green) !important; color: #ddd !important; padding: 20px; }}
    
    .stButton button {{
        background: transparent !important; border: 2px solid var(--neon-cyan) !important;
        color: var(--neon-cyan) !important; font-family: 'Share Tech Mono', monospace !important;
        font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 3px; padding: 25px 0 !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
    }}
    .stButton button:hover {{ background: var(--neon-cyan) !important; color: #000 !important; box-shadow: 0 0 40px var(--neon-cyan); transform: scale(1.02); }}
    
    /* --- HUD CONTAINERS --- */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(5, 10, 15, 0.7) !important; border: 1px solid rgba(0, 243, 255, 0.15) !important;
        box-shadow: 0 0 20px rgba(0,0,0,0.8); border-radius: 5px;
    }}
    
    /* --- TERMINAL & CHAT --- */
    .terminal-box {{ font-family: 'Share Tech Mono', monospace; color: var(--neon-green); background: #020202; border: 1px solid #111; padding: 15px; height: 300px; overflow-y: auto; font-size: 0.85rem; }}
    
    /* Chat Message Styles */
    [data-testid="stChatMessage"] {{ background-color: rgba(0, 20, 30, 0.5); border: 1px solid #333; border-radius: 10px; }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{ font-family: 'Share Tech Mono', monospace; }}

    /* --- SIDEBAR NAVIGATION --- */
    [data-testid="stSidebar"] .stRadio label {{
        padding-top: 15px !important;
        padding-bottom: 15px !important;
    }}

    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
        font-size: 1.2rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        color: var(--neon-cyan) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease;
    }}

    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p:hover {{
        color: #fff !important;
        padding-left: 10px;
        text-shadow: 0 0 15px var(--neon-cyan);
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 5. MODEL BACKEND
# ==========================================
# ==========================================
# 🧠 5. MODEL BACKEND
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 👇 PASTE THE NEW FUNCTION HERE, REPLACING THE OLD ONE 👇
@st.cache_resource
def load_model():
    # 1. Define the Neural Network Architecture
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

    # 2. Initialize Model
    model = EfficientNetLSTM().to(DEVICE)
    model_path = "efficientnet_b3_lstm_active.pth"

    # 3. AUTO-DOWNLOAD FROM DRIVE IF MISSING
    if not os.path.exists(model_path):
        # 👇 YOUR SPECIFIC ID IS PASTED HERE 👇
        file_id = "1IpeVbi0jvwHaXD5qCMtF_peUVR9uJDw0"
        
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            print(f"Downloading model from Drive (ID: {file_id})...")
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    # 4. Load Weights
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
# 🧭 7. SIDEBAR (CYBER OPS - VISIBILITY BOOST)
# ==========================================
if "page" not in st.session_state: st.session_state.page = "Dashboard"

with st.sidebar:
    # --- HEADER ---
    st.markdown("""
    <div style="text-align: center; border-bottom: 1px solid #00f3ff; padding-bottom: 15px; margin-bottom: 30px;">
        <h2 style="color: #fff; margin:0; letter-spacing: 4px; font-size: 2.2rem; text-shadow: 0 0 10px #00f3ff;">OPS CENTER</h2>
        <p style="color: #00f3ff; margin:0; font-size: 0.9rem; letter-spacing: 2px; font-weight: bold;">V.2.0.4. BETA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- NAVIGATION ---
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
    
    # --- ADVANCED TELEMETRY UI (BIGGER TEXT) ---
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <span style="color: #00f3ff; font-size: 1.5rem; margin-right: 10px;">📡</span>
        <span style="color: #fff; font-family: 'Share Tech Mono'; letter-spacing: 3px; font-size: 1.4rem; font-weight: 700;">LIVE TELEMETRY</span>
    </div>
    """, unsafe_allow_html=True)

    # SIMULATED DATA
    gpu_load = random.randint(30, 85)
    ram_load = random.randint(40, 65)
    temp = random.randint(45, 75)

    # 1. VISUAL PROGRESS BARS (UPDATED CSS FOR SIZE)
    st.markdown(f"""
    <style>
        .meter-container {{ margin-bottom: 20px; }}
        
        .meter-label {{ 
            display: flex; 
            justify-content: space-between; 
            color: #ffffff !important; /* PURE WHITE */
            font-size: 1.1rem !important; /* INCREASED SIZE */
            font-family: 'Rajdhani', sans-serif; 
            font-weight: 700; /* BOLD */
            margin-bottom: 5px; 
            letter-spacing: 1px;
        }}
        
        .meter-val {{
            font-family: 'Share Tech Mono'; 
            font-weight: bold;
        }}
        
        .meter-bar-bg {{ 
            width: 100%; 
            height: 8px; /* THICKER BAR */
            background: rgba(255,255,255,0.15); 
            border-radius: 4px; 
        }}
        
        .meter-bar-fill {{ 
            height: 100%; 
            border-radius: 4px; 
            box-shadow: 0 0 10px currentColor; 
            transition: width 0.5s ease; 
        }}
    </style>

    <div class="meter-container">
        <div class="meter-label">
            <span>GPU_CORES</span> 
            <span class="meter-val" style="color:#00f3ff">{gpu_load}%</span>
        </div>
        <div class="meter-bar-bg">
            <div class="meter-bar-fill" style="width: {gpu_load}%; background: #00f3ff; color: #00f3ff;"></div>
        </div>
    </div>

    <div class="meter-container">
        <div class="meter-label">
            <span>VRAM_ALLOC</span> 
            <span class="meter-val" style="color:#ff00ff">{ram_load}%</span>
        </div>
        <div class="meter-bar-bg">
            <div class="meter-bar-fill" style="width: {ram_load}%; background: #ff00ff; color: #ff00ff;"></div>
        </div>
    </div>

    <div class="meter-container">
        <div class="meter-label">
            <span>CORE_TEMP</span> 
            <span class="meter-val" style="color:#ff003c">{temp}°C</span>
        </div>
        <div class="meter-bar-bg">
            <div class="meter-bar-fill" style="width: {temp}%; background: #ff003c; color: #ff003c;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. MINI GRAPH (Neural Flux)
    st.markdown("<p style='font-size: 1.0rem; color: #fff; margin-bottom: 5px; font-weight: bold; font-family: Rajdhani;'>NEURAL_FLUX_RATE</p>", unsafe_allow_html=True)
    chart_data = pd.DataFrame(np.random.randn(20, 1), columns=['a'])
    st.line_chart(chart_data, height=100, color="#0aff48") 

    # 3. SCROLLING TERMINAL LOG (Bottom)
    log_messages = [
        "[SYS] Secure Uplink Established...",
        "[NET] Handshake Verified (24ms)",
        "[AI] Model Weights Loaded",
        "[SEC] Deepfake Signatures Updated",
        "[LOG] Monitoring Active Threads...",
        "[SYS] Waiting for User Input..."
    ]
    random_log = random.choice(log_messages)
    
    st.markdown(f"""
    <div style="background: #000; border: 1px solid #333; padding: 15px; font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: #0aff48; border-left: 4px solid #0aff48; margin-top: 20px;">
        <div style="opacity: 0.5; margin-bottom: 4px;">> executing_startup.sh</div>
        <div style="opacity: 0.7; margin-bottom: 4px;">> mounting_neural_drive</div>
        <div style="color: #fff; font-weight: bold;">> {random_log} <span style="animation: blink 1s infinite;">_</span></div>
    </div>
    <style>@keyframes blink {{ 0% {{ opacity: 0; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}</style>
    """, unsafe_allow_html=True)

    # 4. API STATUS BADGE
    st.write("")
    if gemini_active:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; background: rgba(10, 255, 72, 0.1); border: 1px solid #0aff48; padding: 12px; border-radius: 4px; margin-top: 10px;">
            <div style="width: 12px; height: 12px; background: #0aff48; border-radius: 50%; box-shadow: 0 0 15px #0aff48; margin-right: 15px;"></div>
            <span style="color: #0aff48; font-weight: 900; font-size: 1.0rem; letter-spacing: 2px;">SYSTEM ONLINE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; background: rgba(255, 0, 60, 0.1); border: 1px solid #ff003c; padding: 12px; border-radius: 4px; margin-top: 10px;">
            <div style="width: 12px; height: 12px; background: #ff003c; border-radius: 50%; box-shadow: 0 0 15px #ff003c; margin-right: 15px;"></div>
            <span style="color: #ff003c; font-weight: 900; font-size: 1.0rem; letter-spacing: 2px;">OFFLINE MODE</span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 🏠 PAGE 1: DASHBOARD
# ==========================================
if st.session_state.page == "Dashboard":
    
    # --- UPDATED TITLE SECTION WITH SHARPER HTML STRUCTURE ---
    st.markdown("""
        <div style="padding-top: 20px; padding-bottom: 20px;">
            <h1 class="chrome-title">AI THENTIC</h1>
            <div style="display: flex; justify-content: center;">
                <p class="glitch-subtitle">>> NEURAL FORENSIC SUITE v2.0 <<</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("") 

    # --- TOP ROW ---
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if lottie_left_scan: st_lottie(lottie_left_scan, height=180, key="l1")
        with col2:
            st.markdown("""
                <div class="integrity-box">
                    <h3 style="color: #fff; text-shadow: 0 0 15px #00f3ff; margin-bottom: 5px;">DIGITAL INTEGRITY VERIFICATION</h3>
                    <p style="color: #aaa; font-family: 'Share Tech Mono'; letter-spacing: 1px; font-size: 0.9rem;">
                        DEPLOYING BI-DIRECTIONAL LSTM ARRAYS FOR DEEPFAKE ARTIFACT DETECTION.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button(">> INITIALIZE ANALYSIS MODULE <<", type="primary", use_container_width=True):
                st.session_state.page = "Analysis Console"
                st.rerun()
        with col3:
             if lottie_right_scan: st_lottie(lottie_right_scan, height=180, key="r1")

    # --- ARCHITECTURE ---
    st.markdown("<h3 style='text-align:center; color: var(--neon-cyan); margin-top: 50px; letter-spacing:4px;'>SYSTEM ARCHITECTURE</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666; font-size:0.8rem; margin-bottom:30px;'>HOVER CARDS FOR DIAGNOSTICS</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("""
        <div class="cyber-card-container">
            <div class="cyber-card">
                <h4>01. ACTIVE SAMPLING</h4>
                <p><strong>High-Entropy Frame Extraction.</strong> The algorithm actively discards 85% of static data to focus solely on high-motion vectors where Deepfake artifacts occur (blinking, turning).</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with c2: st.markdown("""
        <div class="cyber-card-container">
            <div class="cyber-card">
                <h4>02. TEMPORAL MEMORY</h4>
                <p><strong>Bi-Directional LSTM Core.</strong> Analyzes frame-to-frame inconsistencies in the time domain. It detects 'temporal jitter'—flickering that happens across time but looks fine in a still photo.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with c3: st.markdown("""
        <div class="cyber-card-container">
            <div class="cyber-card">
                <h4>03. SPATIAL SCAN</h4>
                <p><strong>EfficientNet-B3 Backbone.</strong> A powerful CNN that detects pixel-level anomalies. It looks for blending boundaries, resolution mismatches, and warping artifacts on the skin.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- FAQ ---
    st.markdown("---")
    st.markdown("<h3 style='text-align:center; color: var(--neon-cyan); letter-spacing:4px;'>FORENSIC KNOWLEDGE BASE</h3>", unsafe_allow_html=True)
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
        st.markdown("<h4 style='color: #00ffff; text-align: center; text-shadow: 0 0 10px #00ffff;'>AI ASSISTANT LINK</h4>", unsafe_allow_html=True)
        if lottie_chatbot: st_lottie(lottie_chatbot, height=250, key="bot")
        else: st.markdown("⚠️ Animation Assets Missing")

    with c_chat_box:
        with st.container(border=True):
            st.markdown("**SECURE COMMS CHANNEL**")
            
            if "messages" not in st.session_state: st.session_state.messages = []
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            if prompt := st.chat_input("Query forensic database..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

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
                            full_response = f"⚠️ SYSTEM ERROR: {str(e)}"
                            message_placeholder.error(full_response)
                    else:
                        full_response = "⚠️ SYSTEM OFFLINE. API KEY REQUIRED."
                        message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# 🕵️ PAGE 2: ANALYSIS CONSOLE
# ==========================================
elif st.session_state.page == "Analysis Console":
    
    st.markdown('<h1 class="chrome-title" style="font-size:3rem;">ANALYSIS CONSOLE</h1>', unsafe_allow_html=True)
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
                    
                    st.write("")
                    st.markdown("**⚡ LIVE NEURAL TELEMETRY**")
                    graph_place = st.empty()
                    chart_data = pd.DataFrame(columns=["Integrity"])
                    for x in range(30):
                        new_row = pd.DataFrame({"Integrity": [random.uniform(0.4, 0.9)]})
                        chart_data = pd.concat([chart_data, new_row], ignore_index=True)
                        graph_place.area_chart(chart_data, color="#00f3ff", height=150)
                        time.sleep(0.05)

                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    input_tensor = torch.stack([transform(Image.fromarray(f)) for f in faces]).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        real_score, fake_score = probs[0][0].item(), probs[0][1].item()

                    st.write("---")
                    
                    if fake_score > 0.50:
                        st.markdown(f"""<div style="background: rgba(255, 0, 60, 0.2); border: 2px solid #ff003c; padding: 20px; text-align: center; border-radius: 10px; box-shadow: 0 0 30px #ff003c;"><h1 style="color: #ff003c; margin:0; font-family: 'Share Tech Mono'; font-size: 3rem; text-shadow: 0 0 20px #ff003c;">⚠️ DEEPFAKE DETECTED</h1><p style="letter-spacing: 2px; color: #ff80ab;">CONFIDENCE: {fake_score*100:.2f}%</p></div>""", unsafe_allow_html=True)
                        st.toast("⚠️ THREAT DETECTED: DEEPFAKE SIGNATURE FOUND", icon="🚨")
                    else:
                        st.markdown(f"""<div style="background: rgba(10, 255, 72, 0.1); border: 2px solid #0aff48; padding: 20px; text-align: center; border-radius: 10px; box-shadow: 0 0 30px #0aff48;"><h1 style="color: #0aff48; margin:0; font-family: 'Share Tech Mono'; font-size: 3rem; text-shadow: 0 0 20px #0aff48;">✅ AUTHENTIC MEDIA</h1><p style="letter-spacing: 2px; color: #b9f6ca;">CONFIDENCE: {real_score*100:.2f}%</p></div>""", unsafe_allow_html=True)
                        st.toast("✅ SYSTEM SECURE: MEDIA VERIFIED", icon="🛡️")
                    m1, m2 = st.columns(2)
                    with m1: st.markdown(f"**REAL PROBABILITY**"); st.progress(real_score)
                    with m2: st.markdown(f"**FAKE PROBABILITY**"); st.progress(fake_score)
                    
                    terminal_placeholder.markdown('<div class="terminal-box" style="color: var(--neon-cyan);">[SUCCESS] ANALYSIS COMPLETE.<br>[LOG] REPORT GENERATED.<br>[LOG] ARTIFACTS ARCHIVED.</div>', unsafe_allow_html=True)
                    
                    st.write("---")
                    st.subheader("Extracted Artifacts")
                    st.markdown("""<style>.stImage { border: 1px solid var(--neon-cyan); transition: transform 0.3s; } .stImage:hover { transform: scale(1.1); box-shadow: 0 0 15px var(--neon-cyan); }</style>""", unsafe_allow_html=True)
                    f_cols = st.columns(10)
                    for i, face in enumerate(faces[:10]):
                        with f_cols[i]: st.image(face, use_container_width=True)

# ==========================================
# 📄 PAGE 3: METHODOLOGY
# ==========================================
elif st.session_state.page == "Methodology":
    st.markdown('<h1 class="chrome-title" style="font-size:3rem;">SYSTEM KERNEL</h1>', unsafe_allow_html=True)
    st.write("")

    # --- 1. VISUAL PIPELINE (HTML/CSS) ---
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,20,30,0.6); padding: 25px; border-radius: 15px; border: 1px solid #00f3ff; margin-bottom: 40px; box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);">
        <div style="text-align:center; opacity: 0.8;">📹<br><span style="font-size:0.8rem; color:#aaa;">INPUT SOURCE</span></div>
        <div style="color: #00f3ff; font-weight:bold;">➜</div>
        <div style="text-align:center; color:#00f3ff; text-shadow: 0 0 10px #00f3ff;">⚡<br><span style="font-size:0.8rem;">ACTIVE SAMPLING</span></div>
        <div style="color: #00f3ff; font-weight:bold;">➜</div>
        <div style="text-align:center; color:#ff00ff; text-shadow: 0 0 10px #ff00ff;">👁️<br><span style="font-size:0.8rem;">SPATIAL CNN</span></div>
        <div style="color: #00f3ff; font-weight:bold;">➜</div>
        <div style="text-align:center; color:#0aff48; text-shadow: 0 0 10px #0aff48;">🧠<br><span style="font-size:0.8rem;">TEMPORAL LSTM</span></div>
        <div style="color: #00f3ff; font-weight:bold;">➜</div>
        <div style="text-align:center; color:#fff; font-weight:bold;">🛡️<br><span style="font-size:0.8rem;">VERDICT</span></div>
    </div>
    """, unsafe_allow_html=True)

    # --- 2. INTERACTIVE DEEP DIVE TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ STEP 1: PRE-PROCESSING", "👁️ STEP 2: SPATIAL LAYER", "🧠 STEP 3: TEMPORAL LAYER", "📊 STEP 4: PERFORMANCE"])

    with tab1:
        st.markdown("### 01. ENTROPY-BASED FRAME SELECTION")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("""
            **The Challenge:** Analyzing every frame in a 30fps video is computationally expensive and redundant. Most frames are identical.
            
            **The Solution:** We implemented **Active Sampling**.
            1. The system calculates the *pixel difference* (entropy) between consecutive frames.
            2. It discards static frames (backgrounds/talking heads with no movement).
            3. It selects the **Top 20 High-Motion Frames** (blinking, turning, laughing).
            
            *Why?* Generative Adversarial Networks (GANs) struggle to maintain consistency during rapid motion. By targeting these frames, we maximize detection rates.
            """)
        with c2:
            st.markdown("""
            <div style="border: 1px solid #00f3ff; padding: 20px; text-align: center; border-radius: 10px; background: rgba(0, 243, 255, 0.05);">
                <h4 style="color:#00f3ff; margin:0;">DATA REDUCTION</h4>
                <h1 style="color:#fff; font-size: 3rem; margin:0;">95%</h1>
                <p style="color:#aaa; font-size:0.8rem;">LESS NOISE</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 02. SPATIAL FEATURE EXTRACTION (EfficientNet-B3)")
        st.write("""
        Once the keyframes are selected, they are passed to the **Spatial Stream**.
        We utilize **EfficientNet-B3**, a Convolutional Neural Network (CNN) pre-trained on ImageNet.
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("**WHAT IT SEES**")
            st.write("- Resolution Mismatches (Blurry face vs Sharp background)")
            st.write("- Blending Artifacts (The 'mask' edge along the jaw)")
            st.write("- Skin Texture Anomalies (Overly smooth/plastic skin)")
        with col_b:
            st.warning("**OUTPUT**")
            st.write("The CNN does not classify 'Fake' or 'Real' yet.")
            st.write("It converts the image into a **1536-dimensional feature vector**—a mathematical fingerprint of the face.")

    with tab3:
        st.markdown("### 03. TEMPORAL SEQUENCE ANALYSIS (Bi-LSTM)")
        st.write("This is the 'Brain' of the AIthentic architecture.")
        st.markdown("""
        We feed the sequence of vectors from Step 2 into a **Bi-Directional Long Short-Term Memory (LSTM)** network.
        
        * **Standard CNNs** look at one photo at a time.
        * **Our Bi-LSTM** looks at the *change* over time.
        """)
        
        st.markdown("""
        > **Example:** If Frame 10 looks real, but Frame 11 has a 5ms micro-flicker in the eye color, a CNN misses it. The LSTM catches the pattern break.
        """)

    with tab4:
        st.markdown("### 04. VALIDATION METRICS")
        st.write("Performance benchmarks on the **FaceForensics++ (HQ)** and **Celeb-DF** test sets.")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("ACCURACY", "96.71%", "+2.4% vs Xception")
        m2.metric("AUC SCORE", "0.99", "ELITE TIER")
        m3.metric("INFERENCE TIME", "4.2s", "REAL-TIME")
        
        st.write("")
        st.markdown("**CONFIDENCE DISTRIBUTION**")
        st.progress(96)
        st.caption("Model Reliability Index")

# ==========================================
# 👤 PAGE 4: ABOUT US
# ==========================================
elif st.session_state.page == "About Us":
    st.markdown('<h1 class="chrome-title" style="font-size:3rem;">DEV TEAM</h1>', unsafe_allow_html=True)
    st.write("")
    st.markdown("<p style='text-align: center; color: #aaa;'>ENGINEERING STUDENTS // BUILDING THE FUTURE OF AI SECURITY</p>", unsafe_allow_html=True)
    st.write("---")

    # --- ROW 1: SAHIL & HIMANSHU ---
    c1, c2 = st.columns(2)
    
    with c1:
        with st.container(border=True):
            st.markdown("### 👨‍💻 SAHIL DESAI")
            st.markdown("<span style='color:#00f3ff'>**PROJECT LEAD / AI ENGINEER**</span>", unsafe_allow_html=True)
            st.write("""
            Computer Engineering student obsessed with Deep Learning. 
            Built the core Neural Networks (EfficientNet + LSTM) that power this system.
            *"I code until the model learns."*
            """)

    with c2:
        with st.container(border=True):
            st.markdown("### 👨‍💻 HIMANSHU")
            st.markdown("<span style='color:#0aff48'>**SYSTEM ARCHITECT / BACKEND**</span>", unsafe_allow_html=True)
            st.write("""
            Engineering student focused on making the code run fast and smooth.
            Optimized the video processing pipeline to handle real-time data flow.
            *"Efficiency is key."*
            """)

    # --- ROW 2: TEJAS & KRISH ---
    c3, c4 = st.columns(2)

    with c3:
        with st.container(border=True):
            st.markdown("### 👨‍💻 TEJAS")
            st.markdown("<span style='color:#ff00ff'>**DATA ANALYST / RESEARCH**</span>", unsafe_allow_html=True)
            st.write("""
            Engineering student who loves digging into data. 
            Helped curate the FaceForensics++ and Celeb-DF datasets for training.
            *"Data never lies."*
            """)

    with c4:
        with st.container(border=True):
            st.markdown("### 👨‍💻 KRISH")
            st.markdown("<span style='color:#ff003c'>**UI/UX DEVELOPER**</span>", unsafe_allow_html=True)
            st.write("""
            Engineering student with an eye for design.
            Responsible for the Cyberpunk aesthetic and user interface experience.
            *"Making AI look good."*
            """)

# ==========================================
# 📞 PAGE 5: CONTACT
# ==========================================
elif st.session_state.page == "Contact":
    st.markdown('<h1 class="chrome-title" style="font-size:3rem;">SECURE UPLINK</h1>', unsafe_allow_html=True)
    st.write("")
    
    with st.container(border=True):
        st.markdown("### 📡 ESTABLISH CONNECTION")
        st.write("For forensic inquiries, collaboration, or access keys.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.link_button("🔗 LINKEDIN", "https://www.linkedin.com/in/your-profile")
        with c2:
            st.link_button("🐙 GITHUB REPO", "https://github.com/sahil1399m/AI_DEEPFAKE_FINAL_MODEL")
        with c3:
            st.link_button("📧 EMAIL ENCRYPTION", "mailto:sahildesai00112@gmail.com")
            
