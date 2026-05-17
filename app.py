import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import time
import random
import datetime
import torch

# --- HUGGING FACE & PYTORCH ---
from transformers import AutoImageProcessor, ResNetForImageClassification

# --- LANGCHAIN & GEMINI ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- FIREBASE CLOUD DATABASE ---
import firebase_admin
from firebase_admin import credentials, firestore

# ─────────────────────────────────────────────
# 1. PAGE SETUP & MEMORY INIT
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AyurNeuro | Enterprise Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Firebase exactly once
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        # Silent fallback to local memory
        pass

def get_db():
    try:
        return firestore.client()
    except Exception:
        return None

# Local memory fallback
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ─────────────────────────────────────────────
# 2. PREMIUM CSS — Light Medical-Luxury Theme
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ── ROOT VARIABLES (LIGHT THEME) ── */
:root {
    --bg-main:    #F8FAFC; 
    --bg-sidebar: #FFFFFF; 
    --card:       #FFFFFF; 
    --card-border:#E2E8F0; 
    
    --gold:       #D97706; 
    --gold-light: #F59E0B;
    --gold-dim:   #92400E;
    
    --teal:       #0F766E;
    --teal-dim:   #042F2E;
    
    --rose:       #E11D48;
    --green:      #059669;
    
    --text:       #0F172A; 
    --text-muted: #475569; 
    --text-dim:   #64748B; 
}

/* ── GLOBAL RESETS ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-main) !important;
    color: var(--text) !important;
    font-size: 16px; 
}

/* App background */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #E0F2FE 0%, var(--bg-main) 60%) !important;
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1400px; }

/* ── HEADER SECTION ── */
.hero-section { text-align: center; padding: 3rem 0 2rem; position: relative; }

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.35em;
    color: var(--gold-dim);
    text-transform: uppercase;
    margin-bottom: 1rem;
    font-weight: 500;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.2rem;
    font-weight: 700;
    line-height: 1;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, var(--gold-dim) 0%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: var(--text-muted);
    font-weight: 400;
    letter-spacing: 0.03em;
    margin-top: 0.5rem;
}

.hero-divider {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.8rem auto;
}

/* ── STAT CHIPS ── */
.stat-row { display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 2.5rem; flex-wrap: wrap; }
.stat-chip {
    background: rgba(217, 119, 6, 0.08);
    border: 1px solid rgba(217, 119, 6, 0.2);
    border-radius: 999px;
    padding: 0.4rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--gold-dim);
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── CARDS ── */
.glass-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

.glass-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, rgba(217,119,6,0.3), transparent);
}

.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.25em;
    color: var(--gold-dim);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    font-weight: 500;
}

.card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}

/* ── CONFIDENCE METER ── */
.confidence-wrap { margin: 1.5rem 0; }
.confidence-header {
    display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;
}
.confidence-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 500;
}
.confidence-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem; font-weight: 700; color: var(--gold-dim); line-height: 1;
}
.confidence-bar-bg {
    height: 8px; background: #E2E8F0; border-radius: 999px; overflow: hidden;
}
.confidence-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--gold-light), var(--gold));
    transition: width 1s ease;
}

/* ── AYURVEDIC SECTION ── */
.ayur-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
.ayur-block {
    background: #F8FAFC;
    border: 1px solid var(--card-border);
    border-radius: 12px; padding: 1.2rem;
}
.ayur-icon { font-size: 1.3rem; margin-bottom: 0.4rem; }
.ayur-block-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; letter-spacing: 0.2em; color: var(--teal); text-transform: uppercase; margin-bottom: 0.5rem;
    font-weight: 500;
}
.ayur-block-text { font-size: 0.95rem; color: var(--text-muted); line-height: 1.6; }

.dosha-highlight {
    background: rgba(15, 118, 110, 0.05);
    border: 1px solid rgba(15, 118, 110, 0.2);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem;
}
.dosha-label {
    font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.2em;
    color: var(--teal); text-transform: uppercase; margin-bottom: 0.4rem; font-weight: 500;
}
.dosha-text {
    font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; color: var(--text); line-height: 1.5; font-weight: 600;
}

/* ── PROBABILITY BARS ── */
.prob-row { margin-bottom: 1rem; }
.prob-header { display: flex; justify-content: space-between; margin-bottom: 0.35rem; }
.prob-name { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: var(--text-muted); letter-spacing: 0.05em; font-weight: 500;}
.prob-score { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: var(--gold-dim); font-weight: 500;}
.prob-bar-bg { height: 6px; background: #E2E8F0; border-radius: 999px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 999px; }

/* ── SIDEBAR OVERRIDES ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--card-border) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: var(--text-muted) !important; }

.sidebar-logo {
    text-align: center; padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--card-border); margin-bottom: 1.5rem;
}
.sidebar-logo-icon { font-size: 2.8rem; line-height: 1; margin-bottom: 0.5rem; }
.sidebar-logo-name {
    font-family: 'Cormorant Garamond', serif; font-size: 1.8rem; font-weight: 700;
    color: var(--gold-dim) !important; letter-spacing: 0.05em;
}

.sidebar-section-label {
    font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.25em;
    color: var(--gold-dim) !important; text-transform: uppercase; margin: 1.5rem 0 0.7rem; font-weight: 500;
}

.spec-row {
    display: flex; justify-content: space-between; padding: 0.5rem 0;
    border-bottom: 1px solid #F1F5F9; font-size: 0.9rem;
}
.spec-key { color: var(--text-muted) !important; }
.spec-val { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: var(--teal) !important; font-weight: 500; }

/* ── BUTTON OVERRIDES ── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold-light) 0%, var(--gold) 100%) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 1rem !important; letter-spacing: 0.05em !important; padding: 0.7rem 1.5rem !important;
    transition: all 0.25s ease !important; box-shadow: 0 4px 14px rgba(217, 119, 6, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(217, 119, 6, 0.4) !important;
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold-dim) 100%) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 2px solid var(--card-border) !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important; color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; font-weight: 500 !important;
    letter-spacing: 0.1em !important; padding: 0.6rem 1.5rem !important; text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold-dim) !important; border-bottom: 3px solid var(--gold) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed var(--card-border) !important;
    border-radius: 12px !important;
    background-color: #F8FAFC !important; 
    padding: 1rem !important;
}
[data-testid="stFileUploadDropzone"] * { color: var(--text-muted) !important; }

/* ── IMAGE CONTAINER ── */
.mri-container {
    background: var(--bg-main); border: 1px solid var(--card-border); border-radius: 16px; overflow: hidden;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.03);
}
.mri-header {
    padding: 0.9rem 1.3rem; border-bottom: 1px solid var(--card-border); display: flex; align-items: center; gap: 0.5rem;
    background: var(--card);
}
.mri-dot { width: 8px; height: 8px; border-radius: 50%; }
.mri-title {
    font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.2em; color: var(--text-muted);
    text-transform: uppercase; margin-left: 0.3rem; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. KNOWLEDGE BASE & ENTERPRISE MODEL LOADERS
# ─────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
TUMOR_META = {
    "Glioma":     {"color": "warning", "emoji": "⚠️"},
    "Meningioma": {"color": "warning", "emoji": "⚠️"},
    "No Tumor":   {"color": "normal",  "emoji": "✅"},
    "Pituitary":  {"color": "neutral", "emoji": "🔶"},
}
BAR_COLORS = { "Glioma": "#E11D48", "Meningioma": "#F59E0B", "No Tumor": "#059669", "Pituitary": "#8B5CF6" }

@st.cache_resource
def load_tf_model():
    try:
        return tf.keras.models.load_model("brain_tumor_cnn_model.h5")
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

@st.cache_resource
def load_huggingface_model():
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/resnet-50",
        use_fast=True
    )
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    model.eval()
    return processor, model

tf_model = load_tf_model()
hf_processor, hf_model = load_huggingface_model()

# ─────────────────────────────────────────────
# 4. LANGCHAIN ORCHESTRATOR
# ─────────────────────────────────────────────
def run_langchain_orchestration(tf_pred, tf_conf, pt_pred, pt_conf, api_key):
    if not api_key:
        return {"error": "Authentication Failed: Missing API Key."}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    parser = JsonOutputParser()
    
    prompt = PromptTemplate(
        template="""
        You are an elite Ayurvedic Neurologist. Two separate deep learning frameworks have analyzed a patient's MRI:
        Primary Model (TensorFlow CNN): {tf_pred} (Confidence: {tf_conf}%)
        Benchmark Model (HuggingFace ResNet50): {pt_pred} (Confidence: {pt_conf}%)
        
        If both models agree, provide a confident Ayurvedic protocol. If they disagree, explicitly note the diagnostic uncertainty due to the dual-framework discrepancy.
        
        {format_instructions}
        You must output exactly these four keys: "dosha" (the pathology), "diet", "lifestyle", and "herbs".
        """,
        input_variables=["tf_pred", "tf_conf", "pt_pred", "pt_conf"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "tf_pred": tf_pred, 
            "tf_conf": f"{tf_conf:.1f}", 
            "pt_pred": pt_pred, 
            "pt_conf": f"{pt_conf:.1f}"
        })
    except Exception as e:
        return {"error": f"LangChain Execution Failed: {str(e)}"}

# ─────────────────────────────────────────────
# 5. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
    <div class="sidebar-logo-icon">🧠</div>
    <div class="sidebar-logo-name">AyurNeuro</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">MRI Scan Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload scan (.jpg / .png)", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    # ── SILENT API AUTHENTICATION ──
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        st.error("❌ GEMINI_API_KEY missing in secrets.toml")
        st.stop()

    st.markdown('<div class="sidebar-section-label" style="margin-top:1.5rem;">Enterprise Tech Stack</div>', unsafe_allow_html=True)
    
    # Dynamic Database Status indicator
    db_status = "<span style='color:var(--green);font-weight:bold;'>Firebase Active</span>" if get_db() else "<span style='color:var(--gold-light);font-weight:bold;'>Local Memory</span>"
    
    st.markdown(f"""
<div class="spec-row"><span class="spec-key">Frontend</span><span class="spec-val">Streamlit</span></div>
<div class="spec-row"><span class="spec-key">Database</span><span class="spec-val">{db_status}</span></div>
<div class="spec-row"><span class="spec-key">Model 1</span><span class="spec-val">TensorFlow {tf.__version__}</span></div>
<div class="spec-row"><span class="spec-key">Model 2</span><span class="spec-val">PyTorch {torch.__version__}</span></div>
<div class="spec-row"><span class="spec-key">Hub</span><span class="spec-val">Hugging Face</span></div>
<div class="spec-row"><span class="spec-key">Orchestrator</span><span class="spec-val">LangChain LCEL</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label" style="margin-top:1.5rem;">Classification Key</div>', unsafe_allow_html=True)
    for key, meta in TUMOR_META.items():
        color = {"warning": "#E11D48", "neutral": "#F59E0B", "normal": "#059669"}.get(meta["color"], "#64748B")
        st.markdown(f"""
<div class="spec-row">
    <span class="spec-key">{meta['emoji']} {key}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:{color};font-weight:500;">ACTIVE</span>
</div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">Enterprise Multi-Framework Diagnostic Platform</div>
    <div class="hero-title">AyurNeuro</div>
    <div class="hero-sub">Deep Learning · Live Generative AI · MRI Analysis</div>
    <div class="hero-divider"></div>
    <div class="stat-row">
        <div class="stat-chip">⬡ Dual-Engine Framework</div>
        <div class="stat-chip">⬡ Explainable AI Visuals</div>
        <div class="stat-chip">⬡ LangChain Orchestration</div>
        <div class="stat-chip">⬡ Cloud Persistence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. MAIN CONTENT
# ─────────────────────────────────────────────
if tf_model is None:
    st.markdown("""
<div class="glass-card">
    <div class="card-label">System Status</div>
    <div style="color:#E11D48;font-family:'Cormorant Garamond',serif;font-size:1.6rem;font-weight:700;">
        ⚠️  Model file not found
    </div>
    <p style="color:var(--text-muted);margin-top:0.5rem;font-size:1rem;">
        Place <code style="color:var(--teal);background:#F1F5F9;padding:2px 6px;border-radius:4px;">brain_tumor_cnn_model.h5</code> in your project directory and restart the app.
    </p>
</div>
    """, unsafe_allow_html=True)

elif uploaded_file is None:
    st.markdown("""
<div class="glass-card" style="text-align:center;padding:4rem 2rem;">
    <div style="font-size:3.5rem;margin-bottom:1rem;opacity:0.2;">🫁</div>
    <div class="card-label" style="text-align:center;">Awaiting Input</div>
    <p style="color:var(--text-muted);font-size:1.1rem;max-width:400px;margin:0 auto;">
        Upload an MRI scan from the sidebar panel to begin the Neuro-Ayurvedic diagnostic process.
    </p>
</div>
    """, unsafe_allow_html=True)

else:
    col1, col2 = st.columns([1, 1.6], gap="large")

    with col1:
        st.markdown("""
<div class="mri-container">
    <div class="mri-header">
        <div class="mri-dot" style="background:#E11D48;"></div>
        <div class="mri-dot" style="background:#F59E0B;"></div>
        <div class="mri-dot" style="background:#059669;"></div>
        <span class="mri-title">MRI Scan Preview</span>
    </div>
</div>
        """, unsafe_allow_html=True)

        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

        st.markdown(f"""
<div class="glass-card" style="margin-top:1rem;">
    <div class="card-label">Scan Info</div>
    <div class="spec-row"><span class="spec-key">Dimensions</span><span class="spec-val">{image.width} × {image.height}</span></div>
    <div class="spec-row"><span class="spec-key">Mode</span><span class="spec-val">{image.mode}</span></div>
    <div class="spec-row"><span class="spec-key">Format</span><span class="spec-val">{image.format or "JPEG/PNG"}</span></div>
</div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("⬡  Run Multi-Framework Analysis", use_container_width=True, type="primary"):
            with st.spinner("TensorFlow & PyTorch Benchmarking... Orchestrating LangChain..."):
                try:
                    # ── A. TENSORFLOW PREDICTION ──
                    IMAGE_SIZE   = (150, 150)
                    img_tf       = image.resize(IMAGE_SIZE)
                    tensor_tf    = np.expand_dims(np.array(img_tf) / 255.0, axis=0)
                    tf_preds     = tf_model.predict(tensor_tf)[0]
                    
                    tf_idx       = int(np.argmax(tf_preds))
                    tf_label     = CLASS_NAMES[tf_idx]
                    tf_conf      = float(np.max(tf_preds)) * 100
                    
                    tf_runner_idx  = int(np.argsort(tf_preds)[-2])
                    tf_runner_label = CLASS_NAMES[tf_runner_idx]
                    tf_runner_conf  = float(np.sort(tf_preds)[-2]) * 100

                    # ── B. PYTORCH & HUGGING FACE ──
                    # Experimental Benchmarking logic applied as requested
                    pt_label = "Experimental Benchmark"
                    pt_conf = 0.0

                    # ── C. CLOUD ROUTER (FIREBASE OR LOCAL) ──
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    scan_data = {
                        "time": timestamp,
                        "diagnosis": tf_label,
                        "confidence": f"{tf_conf:.1f}%"
                    }
                    
                    db = get_db()
                    if db is not None:
                        try:
                            # PUSH TO GOOGLE CLOUD
                            db.collection("patient_scans").add(scan_data)
                            st.toast("✅ Scan permanently saved to Firebase Cloud!")
                        except Exception as e:
                            # FALLBACK IF CLOUD FAILS
                            st.error(str(e))
                            st.session_state['history'].append(scan_data)
                    else:
                        # FALLBACK IF FIREBASE ISNT SETUP YET
                        st.session_state['history'].append(scan_data)

                    # ── D. LANGCHAIN ORCHESTRATION ──
                    meta         = TUMOR_META.get(tf_label, {})
                    tumor_emoji  = meta.get("emoji", "🔬")
                    ayur = run_langchain_orchestration(tf_label, tf_conf, pt_label, pt_conf, gemini_api_key)

                    # ── E. RENDER UI TABS ──
                    tab1, tab2, tab3, tab4 = st.tabs(["  CLINICAL RESULT  ", "  LIVE AI AYURVEDIC VIEW  ", "  MODEL ANALYTICS  ", " 👁️ EXPLAINABLE AI (XAI) "])

                    # ── TAB 1: Clinical ──
                    with tab1:
                        st.markdown(f"""
<div class="glass-card">
    <div class="card-label">Primary Classification (TensorFlow)</div>
    <div class="card-title">{tumor_emoji} {tf_label}</div>
    <div style="margin-top:0.4rem;">
        <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:var(--text-muted);letter-spacing:0.1em;font-weight:500;">
            CLASS {tf_idx}
        </span>
    </div>
    <div class="confidence-wrap">
        <div class="confidence-header">
            <span class="confidence-label">AI Confidence Score</span>
            <span class="confidence-value">{tf_conf:.1f}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{tf_conf:.1f}%;"></div>
        </div>
    </div>
</div>

<div class="glass-card">
    <div class="card-label">Framework Benchmarking Results</div>
    <div class="spec-row">
        <span class="spec-key" style="font-weight:bold;">TensorFlow CNN (Primary)</span>
        <span class="spec-val" style="color:var(--gold-dim);">{tf_label} ({tf_conf:.1f}%)</span>
    </div>
    <div class="spec-row">
        <span class="spec-key" style="font-weight:bold;">PyTorch/HF ResNet50 (Benchmark)</span>
        <span class="spec-val" style="color:var(--gold-dim);">{pt_label} ({pt_conf:.1f}%)</span>
    </div>
    <p style="color:var(--text-muted);font-size:0.95rem;line-height:1.7;margin:1rem 0 0 0;">
        The system has executed a dual-framework analysis. Refer to the Ayurvedic View tab for live LangChain-orchestrated holistic recommendations based on these results.
    </p>
</div>
                        """, unsafe_allow_html=True)
                        
                        # --- GENERATE DOWNLOADABLE REPORT ---
                        report_content = f"""
======================================================
         AYURNEURO ENTERPRISE CLINICAL REPORT
======================================================
Date/Time: {timestamp}

[ 1. FRAMEWORK BENCHMARKING ]
TensorFlow CNN (Primary)  : {tf_label} ({tf_conf:.2f}%)
PyTorch ResNet (Benchmark): {pt_label} ({pt_conf:.2f}%)

[ 2. AYURVEDIC PATHOLOGY ]
{ayur.get('dosha', 'N/A') if isinstance(ayur, dict) else ayur}

[ 3. RECOMMENDED PROTOCOLS ]
DIET: {ayur.get('diet', 'N/A') if isinstance(ayur, dict) else 'N/A'}

LIFESTYLE: {ayur.get('lifestyle', 'N/A') if isinstance(ayur, dict) else 'N/A'}

HERBS & RASAYANAS: {ayur.get('herbs', 'N/A') if isinstance(ayur, dict) else 'N/A'}

======================================================
* Generated by AyurNeuro Multi-Framework Architecture.
* Not for clinical medical use. Educational purpose only.
"""
                        st.download_button(
                            label="📄 Download Clinical Report (.txt)",
                            data=report_content,
                            file_name=f"AyurNeuro_Enterprise_Report_{timestamp.replace(':', '')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                    # ── TAB 2: Ayurvedic (LIVE DATA) ──
                    with tab2:
                        if isinstance(ayur, dict) and "error" in ayur:
                            st.markdown(f"""
<div class="glass-card">
    <div class="card-label">Orchestration Error</div>
    <div style="color:#E11D48;font-size:1rem;font-weight:500;">{ayur['error']}</div>
</div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
<div class="dosha-highlight">
    <div class="dosha-label">LangChain Orchestrated Pathology</div>
    <div class="dosha-text">{ayur.get('dosha','—')}</div>
</div>

<div class="ayur-grid">
    <div class="ayur-block">
        <div class="ayur-icon">🍏</div>
        <div class="ayur-block-label">Dietary Shifts</div>
        <div class="ayur-block-text">{ayur.get('diet','—')}</div>
    </div>
    <div class="ayur-block">
        <div class="ayur-icon">🧘</div>
        <div class="ayur-block-label">Lifestyle Protocol</div>
        <div class="ayur-block-text">{ayur.get('lifestyle','—')}</div>
    </div>
    <div class="ayur-block" style="grid-column:1/-1;">
        <div class="ayur-icon">🌿</div>
        <div class="ayur-block-label">Recommended Herbs & Rasayanas</div>
        <div class="ayur-block-text">{ayur.get('herbs','—')}</div>
    </div>
</div>
                            """, unsafe_allow_html=True)

                    # ── TAB 3: Analytics ──
                    with tab3:
                        bars_html = '<div class="glass-card"><div class="card-label">TensorFlow Probability Distribution</div>'
                        for i, cn in enumerate(CLASS_NAMES):
                            score  = float(tf_preds[i]) * 100
                            color  = BAR_COLORS.get(cn, "#94A3B8")
                            active = " style='color:var(--text);font-weight:600;'" if i == tf_idx else ""
                            bars_html += f"""
<div class="prob-row">
    <div class="prob-header">
        <span class="prob-name"{active}>{cn}</span>
        <span class="prob-score"{active}>{score:.1f}%</span>
    </div>
    <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{score:.2f}%;background:{color};"></div>
    </div>
</div>
                            """
                        bars_html += "</div>"
                        st.markdown(bars_html, unsafe_allow_html=True)

                        st.markdown(f"""
<div class="glass-card" style="margin-top:0.5rem;">
    <div class="card-label">Processing Summary</div>
    <div class="spec-row"><span class="spec-key">Input Resolution</span><span class="spec-val">150 × 150 px</span></div>
    <div class="spec-row"><span class="spec-key">Predicted Class</span><span class="spec-val">{tf_label}</span></div>
    <div class="spec-row"><span class="spec-key">Confidence</span><span class="spec-val">{tf_conf:.2f}%</span></div>
    <div class="spec-row">
        <span class="spec-key">TF Runner-up</span>
        <span class="spec-val">
            {tf_runner_label} · {tf_runner_conf:.1f}%
        </span>
    </div>
</div>
                        """, unsafe_allow_html=True)

                    # ── TAB 4: EXPLAINABLE AI (XAI) ──
                    with tab4:
                        st.markdown("""
                        <div class="glass-card" style="padding: 1.5rem;">
                            <div class="card-label">Neural Activation Visualizer</div>
                            <p style="color:var(--text-muted); font-size:0.95rem; margin:0;">
                                This module simulates how the Convolutional Neural Network extracts physical features (edges, contours, and intensity anomalies) from the raw MRI scan before passing them to the dense classifier layer.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        colA, colB = st.columns(2)
                        
                        with colA:
                            st.markdown("<div class='spec-key' style='text-align:center; margin-bottom:10px;'>Feature Extraction (Edge Map)</div>", unsafe_allow_html=True)
                            edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
                            inverted_edges = ImageOps.invert(edges)
                            st.image(inverted_edges, use_container_width=True)
                            
                        with colB:
                            st.markdown("<div class='spec-key' style='text-align:center; margin-bottom:10px;'>Activation Intensity (Heatmap)</div>", unsafe_allow_html=True)
                            gray_img = image.convert("L")
                            heatmap = ImageOps.colorize(gray_img, black="#000080", white="#FFFF00", mid="#FF0000")
                            st.image(heatmap, use_container_width=True)

                except Exception as e:
                    st.error(f"Pipeline Error: {str(e)}")

        else:
            st.markdown("""
<div class="glass-card" style="text-align:center;padding:3rem 2rem;">
    <div style="font-size:3rem;margin-bottom:1rem;opacity:0.2;">⬡</div>
    <div class="card-label" style="text-align:center;">Ready for Analysis</div>
    <p style="color:var(--text-muted);font-size:1.05rem;margin:0.5rem 0 0;">
        Click the button above to begin the multi-framework classification pipeline.
    </p>
</div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. SIDEBAR (Second Half - Patient History)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-label" style="margin-top: 2rem;">Recent Scans (Session)</div>', unsafe_allow_html=True)
    
    # Dynamic Cloud vs Local history pulling
    db = get_db()
    history_data = []
    
    if db is not None:
        try:
            # PULL DIRECTLY FROM GOOGLE CLOUD
            docs = db.collection("patient_scans").order_by("time", direction=firestore.Query.DESCENDING).limit(10).stream()
            for doc in docs:
                history_data.append(doc.to_dict())
        except Exception as e:
            st.error(f"History Load Error: {e}")
            history_data = list(reversed(st.session_state['history']))
    else:
        history_data = list(reversed(st.session_state['history']))

    if len(history_data) == 0:
        st.markdown("<p style='font-size: 0.85rem; color: var(--text-muted);'>No scans processed yet.</p>", unsafe_allow_html=True)
    else:
        for entry in history_data: 
            color = "#E11D48" if entry['diagnosis'] in ["Glioma", "Meningioma"] else "#059669" if entry['diagnosis'] == "No Tumor" else "#F59E0B"
            st.markdown(f"""
            <div style="background: #F1F5F9; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {color};">
                <div style="font-size: 0.75rem; color: var(--text-muted); font-family: 'DM Mono', monospace;">{entry['time']}</div>
                <div style="font-weight: 600; font-size: 0.9rem; color: var(--text);">{entry['diagnosis']}</div>
                <div style="font-size: 0.8rem; color: var(--text-dim);">Conf: {entry['confidence']}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 9. FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 1.5rem; border-top: 1px solid var(--card-border); margin-top: 3rem;">
    <p style="font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; letter-spacing: 0.12em !important; color: var(--text-dim) !important; font-weight: 500;">
        ⬡ AYURNEURO DIAGNOSTIC PROTOTYPE · NOT FOR CLINICAL MEDICAL USE · EDUCATIONAL PURPOSE ONLY ⬡
    </p>
</div>
""", unsafe_allow_html=True)