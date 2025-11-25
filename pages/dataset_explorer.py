import streamlit as st
import pandas as pd
from src.models import analyze_text
from src.utils import get_emotion_emoji, load_dataset_tweets, generate_random_tweets

def show_dataset_explorer_page(models):
    st.markdown("<h2 style='font-size: 42px;'>📈 Dataset Explorer</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    df = load_dataset_tweets()
    
    if df is not None:
        st.markdown(f"""
        <div class='prediction-box'>
            <h4 style='font-size: 24px;'>📊 Dataset Information</h4>
            <p style='font-size: 20px !important; color: #e8f0ff !important;'><b style='color: #00d4ff;'>Total Records:</b> {len(df):,}</p>
            <p style='font-size: 20px !important; color: #e8f0ff !important;'><b style='color: #7b2ff7;'>Columns:</b> {', '.join(df.columns.tolist())}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size: 32px;'>📋 Sample Data</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=400)
        
        st.markdown("<h3 style='font-size: 32px;'>📊 Dataset Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h4 style='font-size: 20px;'>Total Rows</h4>
                <p style='font-size: 36px !important; color: #00d4ff !important; font-weight: 700;'>{len(df):,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h4 style='font-size: 20px;'>Total Columns</h4>
                <p style='font-size: 36px !important; color: #7b2ff7 !important; font-weight: 700;'>{len(df.columns)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            text_col = None
            for col in ['text', 'tweet', 'content', 'message', 'Text', 'Tweet']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col:
                avg_len = df[text_col].astype(str).str.len().mean()
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='font-size: 20px;'>Avg Text Length</h4>
                    <p style='font-size: 36px !important; color: #f72585 !important; font-weight: 700;'>{avg_len:.0f}</p>
                    <p style='font-size: 18px !important;'>characters</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='font-size: 20px;'>Text Column</h4>
                    <p style='font-size: 24px !important; color: #f72585 !important;'>Not Found</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size: 32px;'>🔍 Column Analysis</h3>", unsafe_allow_html=True)
        
        for col in df.columns:
            with st.expander(f"📊 {col}", expanded=False):
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.markdown(f"<p style='font-size: 18px !important; color: #e8f0ff !important;'><b>Type:</b> {df[col].dtype}</p>", unsafe_allow_html=True)
                with col_info2:
                    st.markdown(f"<p style='font-size: 18px !important; color: #00d4ff !important;'><b>Non-null:</b> {df[col].notna().sum():,}</p>", unsafe_allow_html=True)
                with col_info3:
                    st.markdown(f"<p style='font-size: 18px !important; color: #f72585 !important;'><b>Null:</b> {df[col].isna().sum():,}</p>", unsafe_allow_html=True)
                
                if df[col].dtype == 'object' and df[col].nunique() < 50:
                    st.markdown("<p style='font-size: 18px !important; color: #e8f0ff !important;'><b>Value Distribution:</b></p>", unsafe_allow_html=True)
                    value_counts = df[col].value_counts().head(10)
                    st.bar_chart(value_counts)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size: 32px;'>🎲 Random Sampling & Quick Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n_samples = st.slider("Number of samples", 1, 10, 3)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            sample_btn = st.button("🎲 Sample & Analyze", use_container_width=True, type="primary")
        
        if sample_btn:
            text_col = None
            for col in ['text', 'tweet', 'content', 'message', 'Text', 'Tweet']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col:
                samples = df[text_col].dropna().astype(str).sample(n_samples).tolist()
                
                for idx, text in enumerate(samples):
                    st.markdown(f"<h4 style='font-size: 24px;'>🔍 Sample {idx + 1}</h4>", unsafe_allow_html=True)
                    
                    with st.spinner(f"Analyzing sample {idx + 1}..."):
                        results = analyze_text(str(text), models)
                        
                        st.markdown(f"""
                        <div class='prediction-box'>
                            <p style='font-size: 18px !important; color: #ffffff !important;'><b>Text:</b> {text}</p>
                            <br>
                            <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
                                <div style='text-align: center; margin: 10px;'>
                                    <p style='font-size: 16px !important; color: #b4c7e7 !important;'>Sarcasm</p>
                                    <p style='font-size: 24px !important; color: {'#f72585' if results['sarcasm']['label'] == 'Sarcastic' else '#00d4ff'} !important; font-weight: 700;'>{results['sarcasm']['label']}</p>
                                    <p style='font-size: 16px !important; color: #e8f0ff !important;'>({results['sarcasm']['confidence']:.1%})</p>
                                </div>
                                <div style='text-align: center; margin: 10px;'>
                                    <p style='font-size: 16px !important; color: #b4c7e7 !important;'>Top Emotion</p>
                                    <p style='font-size: 32px !important;'>{get_emotion_emoji(results['emotions'][0]['label'])}</p>
                                    <p style='font-size: 20px !important; color: #7b2ff7 !important; font-weight: 700;'>{results['emotions'][0]['label'].title()}</p>
                                    <p style='font-size: 16px !important; color: #e8f0ff !important;'>({results['emotions'][0]['score']:.1%})</p>
                                </div>
                                <div style='text-align: center; margin: 10px;'>
                                    <p style='font-size: 16px !important; color: #b4c7e7 !important;'>VADER Sentiment</p>
                                    <p style='font-size: 28px !important; color: {'#00d4ff' if results['vader']['compound'] > 0 else ('#f72585' if results['vader']['compound'] < 0 else '#7b2ff7')} !important; font-weight: 700;'>{results['vader']['compound']:.3f}</p>
                                    <p style='font-size: 16px !important; color: #e8f0ff !important;'>({'Positive' if results['vader']['compound'] > 0 else ('Negative' if results['vader']['compound'] < 0 else 'Neutral')})</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("❌ No text column found in dataset!")
    
    else:
        st.markdown("""
        <div class='prediction-box' style='border-color: rgba(247,37,133,0.5);'>
            <h4 style='font-size: 24px; color: #f72585;'>⚠️ Dataset Not Found</h4>
            <p style='font-size: 18px !important; color: #e8f0ff !important;'>Please ensure your dataset is in the <code>data/</code> folder.</p>
            <br>
            <h4 style='font-size: 20px;'>Expected Files:</h4>
            <ul style='font-size: 18px !important; color: #e8f0ff !important; line-height: 2 !important;'>
                <li><code style='background: rgba(0,212,255,0.2); padding: 5px 10px; border-radius: 5px;'>data/eng_dataset.csv</code></li>
                <li><code style='background: rgba(0,212,255,0.2); padding: 5px 10px; border-radius: 5px;'>data/preprocessed.csv</code></li>
            </ul>
            <br>
            <p style='font-size: 18px !important; color: #e8f0ff !important;'>The dataset should contain a text column (named 'text', 'tweet', 'content', or 'message')</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 24px;'>💡 Using Fallback Sample Tweets</h4>", unsafe_allow_html=True)
        
        if st.button("🎲 Generate Fallback Samples", type="primary"):
            samples = generate_random_tweets(None, 3)
            
            for idx, text in enumerate(samples):
                st.markdown(f"<h4 style='font-size: 22px;'>Sample {idx + 1}</h4>", unsafe_allow_html=True)
                
                with st.spinner(f"Analyzing..."):
                    results = analyze_text(text, st.session_state.models)
                    
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <p style='font-size: 18px !important; color: #ffffff !important;'><b>Text:</b> {text}</p>
                        <br>
                        <div style='display: flex; justify-content: space-around;'>
                            <div style='text-align: center;'>
                                <p style='font-size: 16px !important;'>Sarcasm</p>
                                <p style='font-size: 24px !important; color: {'#f72585' if results['sarcasm']['label'] == 'Sarcastic' else '#00d4ff'} !important; font-weight: 700;'>{results['sarcasm']['label']}</p>
                            </div>
                            <div style='text-align: center;'>
                                <p style='font-size: 16px !important;'>Top Emotion</p>
                                <p style='font-size: 28px !important;'>{get_emotion_emoji(results['emotions'][0]['label'])}</p>
                                <p style='font-size: 18px !important; color: #7b2ff7 !important;'>{results['emotions'][0]['label'].title()}</p>
                            </div>
                            <div style='text-align: center;'>
                                <p style='font-size: 16px !important;'>VADER</p>
                                <p style='font-size: 24px !important; color: {'#00d4ff' if results['vader']['compound'] > 0 else '#f72585'} !important; font-weight: 700;'>{results['vader']['compound']:.3f}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
