import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.models import analyze_text
from src.visualization import create_emotion_chart, create_sarcasm_gauge, create_vader_chart
from src.utils import get_emotion_emoji, load_dataset_tweets, generate_random_tweets
from src.explainability import explain_with_lime, explain_with_shap

def show_explainability_page(models):
    st.markdown("<h2 style='font-size: 42px;'>🔍 Explainability (XAI)</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='prediction-box' style='background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,247,0.1));'>
        <h4 style='font-size: 24px;'>Why did the model predict this?</h4>
        <p style='font-size: 18px !important; color: #e8f0ff !important;'>
            Explainable AI (XAI) helps us understand the "black box" of complex models.
        </p>
        <ul style='font-size: 18px !important; color: #e8f0ff !important; line-height: 2 !important;'>
            <li><b style='color: #00d4ff;'>LIME (Local Interpretable Model-agnostic Explanations):</b> Shows which words contribute most to the prediction</li>
            <li><b style='color: #7b2ff7;'>SHAP (SHapley Additive exPlanations):</b> Provides game-theoretic feature importance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Check if we have tweets from Batch Analysis
    if st.session_state.random_tweets and len(st.session_state.random_tweets) > 0:
        st.markdown("""
        <div class='prediction-box' style='background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,47,247,0.15));'>
            <h4 style='font-size: 22px;'>✅ Using Tweets from Batch Analysis</h4>
            <p style='font-size: 18px !important; color: #e8f0ff !important;'>
                Select any tweet below to see detailed LIME and SHAP explanations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        explainability_tweets = st.session_state.random_tweets
    else:
        st.markdown("""
        <div class='prediction-box' style='border-color: rgba(247,37,133,0.5);'>
            <h4 style='font-size: 22px;'>⚠️ No Tweets from Batch Analysis</h4>
            <p style='font-size: 18px !important; color: #e8f0ff !important;'>
                Please go to <b>📊 Batch Analysis</b> page and generate random tweets first.<br>
                Or click the button below to generate new tweets here.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h3 style='font-size: 28px;'>Generate Tweets for Explanation</h3>", unsafe_allow_html=True)
        with col2:
            if st.button("🎲 Generate Tweets", use_container_width=True):
                df = load_dataset_tweets()
                st.session_state.random_tweets = generate_random_tweets(df, 10)
                st.rerun()
        
        # Use session state or generate new
        if 'random_tweets' in st.session_state and st.session_state.random_tweets:
            explainability_tweets = st.session_state.random_tweets
        else:
            return  # Exit if no tweets available
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 28px;'>📋 Select a Tweet to Explain ({len(explainability_tweets)} tweets available)</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    for idx, tweet in enumerate(explainability_tweets):
        if st.button(f"🎯 Tweet {idx + 1}: {tweet[:80]}{'...' if len(tweet) > 80 else ''}",
                     key=f"tweet_btn_{idx}",
                     use_container_width=True):
            st.session_state.selected_tweet_index = idx
    
    if st.session_state.selected_tweet_index is not None:
        selected_tweet = explainability_tweets[st.session_state.selected_tweet_index]
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='prediction-box' style='background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,47,247,0.15));'>
            <h4 style='font-size: 24px;'>📝 Selected Tweet:</h4>
            <p style='font-size: 22px !important; line-height: 1.8 !important; color: #ffffff !important; font-weight: 500;'>{selected_tweet}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧪 Generating explanations (optimized for speed)..."):
            results = analyze_text(selected_tweet, models)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                label_color = '#f72585' if results['sarcasm']['label'] == 'Sarcastic' else '#00d4ff'
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='font-size: 20px;'>Sarcasm Prediction</h4>
                    <p style='font-size: 36px !important; color: {label_color} !important; font-weight: 700;'>{results['sarcasm']['label']}</p>
                    <p style='font-size: 20px !important;'>Confidence: {results['sarcasm']['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                top_emotion = results['emotions'][0]
                emoji = get_emotion_emoji(top_emotion['label'])
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <div style='font-size: 48px;'>{emoji}</div>
                    <h4 style='font-size: 20px;'>Top Emotion</h4>
                    <p style='font-size: 28px !important; color: #7b2ff7 !important; font-weight: 700;'>{top_emotion['label'].title()}</p>
                    <p style='font-size: 20px !important;'>Score: {top_emotion['score']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                vader_color = '#00d4ff' if results['vader']['compound'] > 0 else ('#f72585' if results['vader']['compound'] < 0 else '#7b2ff7')
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='font-size: 20px;'>VADER Sentiment</h4>
                    <p style='font-size: 36px !important; color: {vader_color} !important; font-weight: 700;'>{results['vader']['compound']:.3f}</p>
                    <p style='font-size: 18px !important;'>{'Positive' if results['vader']['compound'] > 0 else ('Negative' if results['vader']['compound'] < 0 else 'Neutral')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # LIME Explanation
            st.markdown("<h3 style='font-size: 36px;'>🍋 LIME Explanation</h3>", unsafe_allow_html=True)
            st.markdown("""
            <p style='font-size: 18px !important; color: #e8f0ff !important;'>
            LIME highlights words that push the prediction towards <span style='color: #f72585; font-weight: 700;'>"Sarcastic"</span> (red/pink) or <span style='color: #00d4ff; font-weight: 700;'>"Not Sarcastic"</span> (blue/cyan).
            </p>
            """, unsafe_allow_html=True)
            
            try:
                with st.spinner("⚡ Computing LIME (optimized)..."):
                    lime_exp = explain_with_lime(selected_tweet, models)
                    
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
                        height=500,
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
                    
                    st.markdown(f"<h4 style='font-size: 24px;'>📋 Word Importance Table (for {predicted_class_name})</h4>", unsafe_allow_html=True)
                    lime_df = pd.DataFrame(lime_list, columns=['Word/Feature', 'Importance'])
                    
                    if top_label == 1:
                        lime_df['Impact'] = lime_df['Importance'].apply(
                            lambda x: '🔴 Towards Sarcastic' if x > 0 else '🔵 Towards Not Sarcastic'
                        )
                    else:
                        lime_df['Impact'] = lime_df['Importance'].apply(
                            lambda x: '🔵 Towards Not Sarcastic' if x > 0 else '🔴 Towards Sarcastic'
                        )
                    st.dataframe(lime_df, use_container_width=True, height=300)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    try:
                        if lime_list:
                            sorted_lime = sorted(lime_list, key=lambda x: x[1], reverse=True)
                            top_contributors = [word for word, weight in sorted_lime if weight > 0][:2]
                            top_detractors_tuples = [item for item in sorted_lime if item[1] < 0][-2:] 
                            top_detractors = [word for word, weight in top_detractors_tuples]
                            top_detractors.reverse() 
                            
                            text = f"The model predicted **{predicted_class_name}**."
                            if top_contributors:
                                text += f" This was primarily driven by words like **'{top_contributors[0]}'**"
                                if len(top_contributors) > 1:
                                    text += f" and **'{top_contributors[1]}'**."
                                text += " "
                            
                            if top_detractors:
                                other_class = "Sarcastic" if top_label == 0 else "Not Sarcastic"
                                text += f"Words like **'{top_detractors[0]}'**"
                                if len(top_detractors) > 1:
                                    text += f" and **'{top_detractors[1]}'**"
                                text += f" strongly pushed the prediction *towards* **{other_class}** (away from {predicted_class_name})."
                            
                            st.markdown(f"""
                            <div class='prediction-box' style='background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,247,0.1)); border-color: #00d4ff;'>
                                <h4 style='font-size: 22px;'>💡 LIME Interpretation</h4>
                                <p style='font-size: 18px !important; color: #e8f0ff !important;'>{text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not generate LIME interpretation: {e}")
                    
            except Exception as e:
                st.error(f"LIME explanation failed: {str(e)}")
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # SHAP Explanation
            st.markdown("<h3 style='font-size: 36px;'>📊 SHAP Explanation</h3>", unsafe_allow_html=True)
            st.markdown("""
            <p style='font-size: 18px !important; color: #e8f0ff !important;'>
            SHAP values show word contribution using an optimized approximation method.
            </p>
            """, unsafe_allow_html=True)
            
            try:
                with st.spinner("⚡ Computing SHAP (fast approximation)..."):
                    shap_result = explain_with_shap(selected_tweet, models)
                    
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
                            height=450,
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
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class='metric-card' style='text-align: center;'>
                                <h4 style='font-size: 18px;'>Base Value</h4>
                                <p style='font-size: 28px !important; color: #00d4ff !important;'>{shap_result['base_value']:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class='metric-card' style='text-align: center;'>
                                <h4 style='font-size: 18px;'>Final Output</h4>
                                <p style='font-size: 28px !important; color: #7b2ff7 !important;'>{shap_result['final_value']:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            total_impact = sum(values)
                            st.markdown(f"""
                            <div class='metric-card' style='text-align: center;'>
                                <h4 style='font-size: 18px;'>Total Impact</h4>
                                <p style='font-size: 28px !important; color: #f72585 !important;'>{total_impact:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        try:
                            if shap_result:
                                word_scores = list(zip(shap_result['words'], shap_result['scores']))
                                word_scores.sort(key=lambda x: x[1], reverse=True) # Sort descending
                                
                                top_sarcastic_words = [word for word, score in word_scores if score > 0][:2]
                                
                                # Sort for not sarcastic words
                                not_sarcastic_tuples = [(word, score) for word, score in word_scores if score < 0]
                                not_sarcastic_tuples.sort(key=lambda x: x[1]) # Sort ascending by score
                                top_not_sarcastic_words = [word for word, score in not_sarcastic_tuples[:2]]
                                
                                text = "This chart shows each word's impact on the 'Sarcasm' score. Positive values push towards 'Sarcastic', negative values push towards 'Not Sarcastic'.<br><br>"
                                
                                if top_sarcastic_words:
                                    text += f"The words <b style='color: #f72585;'>'{top_sarcastic_words[0]}'</b>"
                                    if len(top_sarcastic_words) > 1:
                                        text += f" and <b style='color: #f72585;'>'{top_sarcastic_words[1]}'</b>"
                                    text += " had the strongest impact pushing the model to predict <b style='color: #f72585;'>Sarcastic</b>. "
                                
                                if top_not_sarcastic_words:
                                    text += f"Conversely, <b style='color: #00d4ff;'>'{top_not_sarcastic_words[0]}'</b>"
                                    if len(top_not_sarcastic_words) > 1:
                                        text += f" and <b style='color: #00d4ff;'>'{top_not_sarcastic_words[1]}'</b>"
                                    text += " had the strongest impact pushing the model to predict <b style='color: #00d4ff;'>Not Sarcastic</b>."

                                st.markdown(f"""
                                <div class='prediction-box' style='background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,247,0.1)); border-color: #7b2ff7;'>
                                    <h4 style='font-size: 22px;'>💡 SHAP Interpretation</h4>
                                    <p style='font-size: 18px !important; color: #e8f0ff !important;'>{text}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Could not generate SHAP interpretation: {e}")
                            
                    else:
                        st.warning("SHAP values could not be computed for this text.")
                        
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {str(e)}")
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='font-size: 36px;'>🔄 LIME vs SHAP Comparison</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='prediction-box'>
                <table style='width:100%; border-collapse: separate; border-spacing: 0 15px;'>
                    <thead>
                        <tr style='background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(123,47,247,0.2));'>
                            <th style='border-bottom: 3px solid #00d4ff; padding: 15px; font-size: 20px !important; color: #ffffff !important;'>Aspect</th>
                            <th style='border-bottom: 3px solid #00d4ff; padding: 15px; font-size: 20px !important; color: #ffffff !important;'>LIME</th>
                            <th style='border-bottom: 3px solid #00d4ff; padding: 15px; font-size: 20px !important; color: #ffffff !important;'>SHAP</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style='padding: 15px; font-size: 18px !important; color: #00d4ff !important; font-weight: 700;'>Approach</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>Local surrogate model</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>Word removal impact</td>
                        </tr>
                        <tr style='background: rgba(255,255,255,0.03);'>
                            <td style='padding: 15px; font-size: 18px !important; color: #00d4ff !important; font-weight: 700;'>Speed</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>Fast (optimized)</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>Very Fast (approximation)</td>
                        </tr>
                        <tr>
                            <td style='padding: 15px; font-size: 18px !important; color: #00d4ff !important; font-weight: 700;'>Interpretability</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>High (simpler explanations)</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>Very High (direct impact)</td>
                        </tr>
                        <tr style='background: rgba(255,255,255,0.03);'>
                            <td style='padding: 15px; font-size: 18px !important; color: #00d4ff !important; font-weight: 700;'>Best For</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>Feature importance ranking</td>
                            <td style='padding: 15px; font-size: 18px !important; color: #e8f0ff !important;'>Understanding word contributions</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
