import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import from our modular source files
from src.models import load_models, analyze_text
from src.visualization import create_emotion_chart, create_sarcasm_gauge, create_vader_chart
from src.utils import get_emotion_emoji, load_dataset_tweets, generate_random_tweets
from src.explainability import explain_with_lime, explain_with_shap

# Import page functions
from pages.home import show_home_page
from pages.single_tweet import show_single_tweet_page
from pages.batch_analysis import show_batch_analysis_page
from pages.explainability import show_explainability_page
from pages.dataset_explorer import show_dataset_explorer_page

# Page config
st.set_page_config(
    page_title="SentiSarc XAI Platform",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for ultra-cool futuristic UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3e 50%, #0f1129 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3e 50%, #0f1129 100%);
    }
    
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #f72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0,212,255,0.5);
        font-weight: 700 !important;
    }
    
    p, li, span, div {
        color: #e8f0ff !important;
        font-family: 'Rajdhani', sans-serif;
        font-size: 18px !important;
        line-height: 1.8 !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid #00d4ff !important;
        border-radius: 15px !important;
        padding: 15px 35px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 25px rgba(102,126,234,0.6) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 0 40px rgba(102,126,234,1), 0 0 60px rgba(123,47,247,0.8) !important;
        border-color: #7b2ff7 !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.08) !important;
        border: 2px solid rgba(0,212,255,0.5) !important;
        border-radius: 20px !important;
        padding: 25px !important;
        backdrop-filter: blur(15px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 0 20px rgba(0,212,255,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02) !important;
        border-color: rgba(123,47,247,0.8) !important;
        box-shadow: 0 15px 50px rgba(123,47,247,0.6), inset 0 0 30px rgba(123,47,247,0.2) !important;
    }
    
    .stTextArea textarea {
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(0,212,255,0.5) !important;
        border-radius: 15px !important;
        color: #000000 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 16px !important;
        padding: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #7b2ff7 !important;
        box-shadow: 0 0 20px rgba(123,47,247,0.6) !important;
    }
    
    /* File uploader styling */
    .stFileUploader label {
        color: #ffffff !important;
    }
    
    .stFileUploader section {
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(0,212,255,0.5) !important;
        border-radius: 15px !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.1) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.9) !important;
        color: #000000 !important;
        border: 2px solid rgba(0,212,255,0.5) !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox input {
        color: #000000 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    /* Dropdown menu items */
    .stSelectbox div[role="listbox"] {
        background: rgba(255,255,255,0.98) !important;
    }
    
    .stSelectbox div[role="option"] {
        color: #000000 !important;
        background: rgba(255,255,255,0.98) !important;
    }
    
    .stSelectbox div[role="option"]:hover {
        background: rgba(0,212,255,0.3) !important;
        color: #000000 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 32px !important;
        color: #00d4ff !important;
        text-shadow: 0 0 15px rgba(0,212,255,0.8);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15)) !important;
        border: 2px solid !important;
        border-image: linear-gradient(135deg, #00d4ff, #7b2ff7) 1 !important;
        border-radius: 20px !important;
        padding: 30px !important;
        margin: 25px 0 !important;
        box-shadow: 0 0 40px rgba(0,212,255,0.3), inset 0 0 30px rgba(0,212,255,0.1) !important;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0,212,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .emoji-large {
        font-size: 56px !important;
        text-align: center;
        margin: 15px 0;
        filter: drop-shadow(0 0 10px rgba(0,212,255,0.6));
    }
    
    .glow-text {
        text-shadow: 0 0 15px rgba(0,212,255,1), 0 0 25px rgba(123,47,247,0.8) !important;
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(102,126,234,0.2);
        border: 2px solid rgba(0,212,255,0.3);
        border-radius: 10px;
        padding: 10px 20px;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0,212,255,0.6);
    }
    
    .stDataFrame {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        overflow: hidden;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d3e 0%, #0a0e27 100%) !important;
        border-right: 2px solid rgba(0,212,255,0.3);
    }
    
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    
    .stRadio label {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(102,126,234,0.2) !important;
        border: 2px solid rgba(0,212,255,0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .tweet-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,247,0.1));
        border: 2px solid rgba(0,212,255,0.4);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .tweet-card:hover {
        transform: translateX(10px);
        border-color: rgba(247,37,133,0.8);
        box-shadow: 0 0 30px rgba(247,37,133,0.5);
        background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,47,247,0.15));
    }
    
    .tweet-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        margin-right: 10px;
        box-shadow: 0 0 15px rgba(102,126,234,0.6);
    }

    [data-testid="stSidebar"] [data-testid="stRadioInput"] {
        border: 2px solid #00d4ff;
        background-color: rgba(0, 212, 255, 0.1);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    [data-testid="stSidebar"] [data-testid="stRadioInput"] div {
        background-color: #f72585 !important;
        box-shadow: 0 0 10px #f72585;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        display: block;
        padding: 8px 10px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0, 212, 255, 0.15);
        transform: translateX(3px);
    }
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div:has(input:checked) label {
        background: rgba(123, 47, 247, 0.2);
        border: 1px solid rgba(123, 47, 247, 0.5);
        font-weight: 700 !important;
    }
    
    /* ===== CUSTOM FIXES ===== */
    /* Remove white space in header */
    .main header {
        visibility: hidden;
    }
    
    /* Remove padding that creates white space */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Make deploy button and menu text BLACK */
    .stDeployButton {
        color: #000000 !important;
    }
    
    .stDeployButton span {
        color: #000000 !important;
    }
    
    .stDeployButton:hover {
        color: #000000 !important;
        background-color: rgba(0,0,0,0.1) !important;
    }
    
    /* Make all menu items in deploy dropdown BLACK */
    [data-testid="stDeployButton"] [role="menu"] {
        background-color: white !important;
    }
    
    [data-testid="stDeployButton"] [role="menu"] * {
        color: #000000 !important;
    }
    
    [data-testid="stDeployButton"] [role="menuitem"] {
        color: #000000 !important;
    }
    
    [data-testid="stDeployButton"] [role="menuitem"]:hover {
        background-color: rgba(0,0,0,0.1) !important;
        color: #000000 !important;
    }
    
    /* Make browser context menu text BLACK */
    .stApp {
        --text-color: #000000 !important;
    }
            
    /* Make the "Browse files" button text BLACK */
    .stFileUploader button {
        color: #000000 !important;
    }
    
    .stFileUploader button span {
        color: #000000 !important;
    }
    
    /* Make ONLY the select dropdown text BLACK */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    /* Make ONLY dropdown menu items BLACK - SUPER TARGETED */
    div[data-baseweb="select"] ~ div[data-baseweb="popover"] li {
        color: #000000 !important;
    }
    
    div[data-baseweb="select"] ~ div[data-baseweb="popover"] li span {
        color: #000000 !important;
    }
    
    /* ===== REMOVE BOTTOM BLACK SPACE ===== */
    .main .block-container {
        padding-bottom: 0rem !important;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stApp {
        min-height: auto !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'random_tweets' not in st.session_state:
    st.session_state.random_tweets = []
if 'selected_tweet_index' not in st.session_state:
    st.session_state.selected_tweet_index = None
if 'single_tweet_text' not in st.session_state:
    st.session_state.single_tweet_text = ""

def main():
    # Header with enhanced styling
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); border-radius: 20px; margin-bottom: 30px; border: 2px solid rgba(0,212,255,0.3);'>
        <h1 style='font-size: 64px; margin-bottom: 10px;'>🎭 SentiSarc XAI Platform</h1>
        <p style='font-size: 24px; color: #00d4ff; font-family: Rajdhani; font-weight: 600; text-shadow: 0 0 20px rgba(0,212,255,0.8);'>
            Advanced Sentiment Analysis • Sarcasm Detection • Explainable AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Remove extra white space
    st.markdown("<style>div.stApp > div:first-child {margin-top: -80px;}</style>", unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; font-size: 28px;'>🎛️ Control Panel</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        page = st.radio(
            "Navigate:",
            ["🏠 Home", "💬 Single Tweet", "📊 Batch Analysis", "🔍 Explainability", "📈 Dataset Explorer"]
        )
        
        st.markdown("---")
        st.markdown("<h3 style='font-size: 22px;'>⚙️ Settings</h3>", unsafe_allow_html=True)
        show_vader = st.checkbox("Show VADER Sentiment", value=True)
        show_emotions = st.checkbox("Show Emotion Distribution", value=True)
        top_n_emotions = st.slider("Top N Emotions", 3, 15, 10)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 15px; background: rgba(102,126,234,0.1); border-radius: 10px; border: 1px solid rgba(0,212,255,0.3);'>
            <p style='font-size: 14px; color: #00d4ff; font-weight: 600;'>
                Powered by GoEmotions & Sarcasm Detection<br>
                🚀 Built with Streamlit<br>
                💎 XAI Enhanced
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("🔮 Loading AI models... This may take a moment..."):
            models = load_models()
            if models:
                st.session_state.models = models
                st.session_state.models_loaded = True
            else:
                st.error("❌ Failed to load models. Please check your installation.")
                return
    else:
        models = st.session_state.models
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "💬 Single Tweet":
        show_single_tweet_page(models, show_vader, show_emotions, top_n_emotions)
    elif page == "📊 Batch Analysis":
        show_batch_analysis_page(models, show_vader, show_emotions)
    elif page == "🔍 Explainability":
        show_explainability_page(models)
    elif page == "📈 Dataset Explorer":
        show_dataset_explorer_page(models)

if __name__ == "__main__":
    main()
