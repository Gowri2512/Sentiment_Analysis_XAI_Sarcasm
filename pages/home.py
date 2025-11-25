import streamlit as st

def show_home_page():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='emoji-large'>🎭</div>
            <h3 style='text-align: center; font-size: 24px;'>Sarcasm Detection</h3>
            <p style='text-align: center; color: #e8f0ff; font-size: 16px !important;'>
                Advanced neural models detect sarcastic content with high accuracy using transformer architecture
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='emoji-large'>😊</div>
            <h3 style='text-align: center; font-size: 24px;'>28 Emotions</h3>
            <p style='text-align: center; color: #e8f0ff; font-size: 16px !important;'>
                Full GoEmotions taxonomy for nuanced emotion classification across the spectrum
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='emoji-large'>🔍</div>
            <h3 style='text-align: center; font-size: 24px;'>XAI Insights</h3>
            <p style='text-align: center; color: #e8f0ff; font-size: 16px !important;'>
                LIME & SHAP explanations for complete transparency and interpretability
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='font-size: 40px; text-align: center;'>✨ Key Features</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(102,126,234,0.1); padding: 25px; border-radius: 15px; border: 2px solid rgba(0,212,255,0.3);'>
            <p style='font-size: 20px !important; line-height: 2 !important;'>
                🎯 <b>Real-time Analysis:</b> Instant predictions on any text<br>
                📊 <b>Batch Processing:</b> Analyze multiple tweets at once<br>
                🎲 <b>Random Sampling:</b> Explore dataset with random tweets<br>
                💾 <b>CSV Upload:</b> Analyze your own datasets
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(118,75,162,0.1); padding: 25px; border-radius: 15px; border: 2px solid rgba(123,47,247,0.3);'>
            <p style='font-size: 20px !important; line-height: 2 !important;'>
                🔬 <b>LIME Explanations:</b> Token-level importance<br>
                📈 <b>SHAP Values:</b> Contribution analysis<br>
                🎨 <b>Interactive Visualizations:</b> Beautiful charts<br>
                🌐 <b>Full Local Deployment:</b> No external APIs needed
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='font-size: 40px; text-align: center;'>🚀 Quick Start Guide</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='prediction-box' style='padding: 40px;'>
        <ol style='font-size: 22px !important; line-height: 2.5 !important; color: #ffffff !important;'>
            <li><b style='color: #00d4ff;'>Select a mode</b> from the sidebar (💬 Single Tweet, 📊 Batch Analysis, etc.)</li>
            <li><b style='color: #7b2ff7;'>Enter text</b> or generate random tweets from dataset</li>
            <li><b style='color: #f72585;'>Click Analyze</b> to see predictions with confidence scores</li>
            <li><b style='color: #00d4ff;'>Explore emotions,</b> sarcasm scores, and sentiment analysis</li>
            <li><b style='color: #7b2ff7;'>Use Explainability page</b> for deep XAI insights with LIME and SHAP</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
