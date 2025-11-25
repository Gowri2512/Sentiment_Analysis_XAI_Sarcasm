import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.models import analyze_text
from src.visualization import create_emotion_chart, create_sarcasm_gauge, create_vader_chart
from src.utils import get_emotion_emoji, load_dataset_tweets, generate_random_tweets
from src.explainability import explain_with_lime, explain_with_shap

def show_single_tweet_page(models, show_vader, show_emotions, top_n_emotions):
    st.markdown("<h2 style='font-size: 42px;'>💬 Single Tweet Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use session state to control the text_area value
        text_input = st.text_area(
            "Enter your text:",
            value=st.session_state.single_tweet_text,
            height=150,
            placeholder="Type or paste a tweet, comment, or any text here..."
        )
        # Update session state with the new value (if user types)
        st.session_state.single_tweet_text = text_input
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Tweet", use_container_width=True):
            df = load_dataset_tweets()
            random_tweets = generate_random_tweets(df, 1)
            if random_tweets:
                st.session_state.single_tweet_text = random_tweets[0]
                st.rerun()
        
        analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    
    if analyze_btn and text_input:
        with st.spinner("🧠 Analyzing..."):
            results = analyze_text(text_input, models)
            
            st.markdown(f"""
            <div class='prediction-box'>
                <h4 style='font-size: 24px;'>📝 Analyzed Text:</h4>
                <p style='font-size: 20px !important; line-height: 1.8 !important; color: #ffffff !important;'>{text_input}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_sarcasm_gauge(
                        results['sarcasm']['confidence'],
                        results['sarcasm']['label']
                    ),
                    use_container_width=True
                )
                
                st.markdown("<h4 style='font-size: 22px;'>Sarcasm Probabilities</h4>", unsafe_allow_html=True)
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric(
                        "Not Sarcastic",
                        f"{results['sarcasm']['prob_not_sarcastic']:.1%}"
                    )
                with prob_col2:
                    st.metric(
                        "Sarcastic",
                        f"{results['sarcasm']['prob_sarcastic']:.1%}"
                    )
            
            with col2:
                if show_emotions:
                    st.plotly_chart(
                        create_emotion_chart(results['emotions'][:top_n_emotions]),
                        use_container_width=True
                    )
            
            if show_vader:
                st.markdown("<h3 style='font-size: 32px;'>📊 VADER Sentiment Analysis</h3>", unsafe_allow_html=True)
                st.plotly_chart(create_vader_chart(results['vader']), use_container_width=True)
            
            st.markdown("<h3 style='font-size: 32px;'>🎭 Detailed Emotion Breakdown</h3>", unsafe_allow_html=True)
            emotion_cols = st.columns(5)
            for idx, emotion in enumerate(results['emotions'][:5]):
                with emotion_cols[idx]:
                    emoji = get_emotion_emoji(emotion['label'])
                    st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <div style='font-size: 48px; margin-bottom: 10px;'>{emoji}</div>
                        <h4 style='font-size: 18px;'>{emotion['label'].title()}</h4>
                        <p style='font-size: 28px !important; color: #00d4ff !important; font-weight: 700;'>{emotion['score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Explainability Section
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 36px;'>🔍 Explainability Analysis</h3>", unsafe_allow_html=True)
            
            # LIME Explanation
            st.markdown("<h4 style='font-size: 28px;'>🍋 LIME Explanation</h4>", unsafe_allow_html=True)
            try:
                with st.spinner("⚡ Computing LIME explanation..."):
                    lime_exp = explain_with_lime(text_input, models)
                    
                    top_label = lime_exp.top_labels[0]
                    lime_list = lime_exp.as_list(label=top_label)
                    predicted_class_name = models['lime_explainer'].class_names[top_label]
                    
                    words = [item[0] for item in lime_list]
                    weights = [item[1] for item in lime_list]
                    
                    if top_label == 1: 
                        colors = ['#f72585' if w > 0 else '#00d4ff' for w in weights]
                        chart_title = "<b>LIME Importance (Towards Sarcastic)</b>"
                        xaxis_title = "Contribution to 'Sarcastic'"
                    else: 
                        colors = ['#00d4ff' if w > 0 else '#f72585' for w in weights]
                        chart_title = "<b>LIME Importance (Towards Not Sarcastic)</b>"
                        xaxis_title = "Contribution to 'Not Sarcastic'"
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=weights,
                            y=words,
                            orientation='h',
                            marker=dict(color=colors, line=dict(color='white', width=2)),
                            text=[f"{w:.3f}" for w in weights],
                            textposition='auto',
                            textfont=dict(size=14, color='white', family='Orbitron')
                        )
                    ])

                    fig.update_layout(
                        title=chart_title,
                        xaxis_title=xaxis_title,
                        yaxis_title="Words/Features",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#ffffff', family='Rajdhani', size=14),
                        height=400,
                        showlegend=False,
                        # SUPER BRIGHT WHITE AXIS LABELS
                        xaxis=dict(
                            title=dict(
                                text=xaxis_title,
                                font=dict(
                                    color='#FFFFFF',
                                    size=22,
                                    family='Orbitron',
                                    weight='bold'
                                )
                            ),
                            tickfont=dict(
                                color='#FFFFFF',
                                size=18,
                                family='Orbitron', 
                                weight='bold'
                            ),
                            title_standoff=25
                        ),
                        yaxis=dict(
                            title=dict(
                                text="Words/Features",
                                font=dict(
                                    color='#FFFFFF',
                                    size=22, 
                                    family='Orbitron',
                                    weight='bold'
                                )
                            ),
                            tickfont=dict(
                                color='#FFFFFF',
                                size=18,
                                family='Orbitron',
                                weight='bold'
                            ),
                            title_standoff=25
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"LIME explanation failed: {str(e)}")
            
            # SHAP Explanation
            st.markdown("<h4 style='font-size: 28px;'>📊 SHAP Explanation</h4>", unsafe_allow_html=True)
            try:
                with st.spinner("⚡ Computing SHAP explanation..."):
                    shap_result = explain_with_shap(text_input, models)
                    
                    if shap_result:
                        words = shap_result['words']
                        values = shap_result['scores']
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(range(len(words))),
                                y=values,
                                text=words,
                                textposition='outside',
                                marker=dict(
                                    color=values,
                                    colorscale='RdBu',
                                    cmid=0,
                                    line=dict(color='white', width=2)
                                ),
                                textfont=dict(size=12, family='Rajdhani'),
                                hovertemplate='<b>%{text}</b><br>Impact: %{y:.4f}<extra></extra>'
                            )
                        ])

                        fig.update_layout(
                            title="<b>SHAP-Style Word Impact</b>",
                            xaxis_title="Word Position",
                            yaxis_title="Impact on Sarcasm Score",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#ffffff', family='Rajdhani', size=14),
                            height=400,
                            showlegend=False,
                            # SUPER BRIGHT WHITE AXIS LABELS
                            xaxis=dict(
                                showticklabels=False,
                                title=dict(
                                    text="Word Position",
                                    font=dict(
                                        color='#FFFFFF',
                                        size=22,
                                        family='Orbitron',
                                        weight='bold'
                                    )
                                )
                            ),
                            yaxis=dict(
                                title=dict(
                                    text="Impact on Sarcasm Score",
                                    font=dict(
                                        color='#FFFFFF',
                                        size=22, 
                                        family='Orbitron',
                                        weight='bold'
                                    )
                                ),
                                tickfont=dict(
                                    color='#FFFFFF',
                                    size=18,
                                    family='Orbitron',
                                    weight='bold'
                                ),
                                title_standoff=25
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {str(e)}")
            
            # Explanation Summary
            st.markdown("<h4 style='font-size: 28px;'>💡 Interpretation Summary</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='prediction-box' style='background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,247,0.1));'>
                <p style='font-size: 18px !important; color: #e8f0ff !important;'>
                The model predicted <b style='color: {'#f72585' if results['sarcasm']['label'] == 'Sarcastic' else '#00d4ff'};'>{results['sarcasm']['label']}</b> with 
                <b>{results['sarcasm']['confidence']:.1%}</b> confidence.
                </p>
                <p style='font-size: 18px !important; color: #e8f0ff !important;'>
                <b>LIME</b> shows which words contribute most to the prediction (red/pink for sarcastic, blue/cyan for not sarcastic).
                </p>
                <p style='font-size: 18px !important; color: #e8f0ff !important;'>
                <b>SHAP</b> shows the impact of each word on the final sarcasm score (positive values push toward sarcastic).
                </p>
            </div>
            """, unsafe_allow_html=True)
