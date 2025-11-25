import streamlit as st
import pandas as pd
from src.models import analyze_text
from src.utils import get_emotion_emoji, load_dataset_tweets, generate_random_tweets

def show_batch_analysis_page(models, show_vader, show_emotions):
    st.markdown("<h2 style='font-size: 42px;'>📊 Batch Tweet Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎲 Random Tweets", "✍️ Custom Tweets", "📁 CSV Upload"])
    
    with tab1:
        st.markdown("<h3 style='font-size: 28px;'>Generate Random Tweets from Dataset</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n_tweets = st.slider("Number of tweets", 5, 20, 10)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 Generate", use_container_width=True, key='gen_random'):
                df = load_dataset_tweets()
                st.session_state.random_tweets = generate_random_tweets(df, n_tweets)
        
        if st.session_state.random_tweets:
            st.markdown(f"<h4 style='font-size: 24px;'>📝 Generated {len(st.session_state.random_tweets)} Tweets</h4>", unsafe_allow_html=True)
            
            for idx, tweet in enumerate(st.session_state.random_tweets):
                st.markdown(f"""
                <div class='tweet-card'>
                    <span class='tweet-number'>Tweet {idx + 1}</span>
                    <p style='font-size: 18px !important; color: #ffffff !important; margin-top: 10px;'>{tweet}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔍 Analyze All Tweets", type="primary", key='analyze_all'):
                progress_bar = st.progress(0)
                st.markdown("<h4 style='font-size: 24px;'>⚡ Analyzing tweets...</h4>", unsafe_allow_html=True)
                
                results_list = []
                
                for idx, tweet in enumerate(st.session_state.random_tweets):
                    results = analyze_text(tweet, models)
                    results_list.append({
                        'Tweet': tweet[:100] + '...' if len(tweet) > 100 else tweet,
                        'Sarcasm': results['sarcasm']['label'],
                        'Confidence': f"{results['sarcasm']['confidence']:.1%}",
                        'Top Emotion': f"{get_emotion_emoji(results['emotions'][0]['label'])} {results['emotions'][0]['label']}",
                        'Emotion Score': f"{results['emotions'][0]['score']:.1%}",
                        'VADER': f"{results['vader']['compound']:.3f}"
                    })
                    progress_bar.progress((idx + 1) / len(st.session_state.random_tweets))
                
                st.markdown("<h3 style='font-size: 32px;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
                results_df = pd.DataFrame(results_list)
                st.dataframe(results_df, use_container_width=True, height=400)
                
                st.markdown("<h4 style='font-size: 24px;'>📈 Summary Statistics</h4>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                sarcastic_count = sum(1 for r in results_list if r['Sarcasm'] == 'Sarcastic')
                with col1:
                    st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <h4 style='font-size: 18px;'>Sarcastic Tweets</h4>
                        <p style='font-size: 32px !important; color: #f72585 !important; font-weight: 700;'>{sarcastic_count}/{len(results_list)}</p>
                        <p style='font-size: 18px !important;'>({sarcastic_count/len(results_list):.1%})</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_conf = sum(float(r['Confidence'].strip('%'))/100 for r in results_list) / len(results_list)
                    st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <h4 style='font-size: 18px;'>Avg Confidence</h4>
                        <p style='font-size: 32px !important; color: #00d4ff !important; font-weight: 700;'>{avg_conf:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_vader = sum(float(r['VADER']) for r in results_list) / len(results_list)
                    vader_color = '#00d4ff' if avg_vader > 0 else ('#f72585' if avg_vader < 0 else '#7b2ff7')
                    st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <h4 style='font-size: 18px;'>Avg VADER</h4>
                        <p style='font-size: 32px !important; color: {vader_color} !important; font-weight: 700;'>{avg_vader:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    emotions = [r['Top Emotion'].split()[-1] for r in results_list]
                    most_common = max(set(emotions), key=emotions.count)
                    emoji = get_emotion_emoji(most_common)
                    st.markdown(f"""
                    <div class='metric-card' style='text-align: center;'>
                        <div style='font-size: 36px;'>{emoji}</div>
                        <h4 style='font-size: 18px;'>Top Emotion</h4>
                        <p style='font-size: 20px !important; color: #7b2ff7 !important; font-weight: 700;'>{most_common.title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add explainability prompt
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class='prediction-box' style='background: linear-gradient(135deg, rgba(123,47,247,0.15), rgba(0,212,255,0.15)); border-color: rgba(123,47,247,0.6);'>
                    <h4 style='font-size: 22px;'>💡 Want to Understand Why?</h4>
                    <p style='font-size: 18px !important; color: #e8f0ff !important;'>
                        Go to <b style='color: #00d4ff;'>🔍 Explainability</b> page to see detailed LIME and SHAP explanations for these tweets!
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3 style='font-size: 28px;'>Enter Multiple Custom Tweets</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 18px !important; color: #e8f0ff !important;'>Enter one tweet per line (max 20 tweets)</p>", unsafe_allow_html=True)
        
        custom_tweets = st.text_area(
            "Enter tweets (one per line):",
            height=300,
            placeholder="Tweet 1\nTweet 2\nTweet 3\n..."
        )
        
        if st.button("🔍 Analyze Custom Tweets", type="primary", key='analyze_custom'):
            tweets = [t.strip() for t in custom_tweets.split('\n') if t.strip()]
            
            if tweets:
                progress_bar = st.progress(0)
                results_list = []
                
                for idx, tweet in enumerate(tweets[:20]):
                    results = analyze_text(tweet, models)
                    results_list.append({
                        'Tweet': tweet[:100] + '...' if len(tweet) > 100 else tweet,
                        'Sarcasm': results['sarcasm']['label'],
                        'Confidence': f"{results['sarcasm']['confidence']:.1%}",
                        'Top Emotion': f"{get_emotion_emoji(results['emotions'][0]['label'])} {results['emotions'][0]['label']}",
                        'Emotion Score': f"{results['emotions'][0]['score']:.1%}",
                        'VADER': f"{results['vader']['compound']:.3f}"
                    })
                    progress_bar.progress((idx + 1) / min(len(tweets), 20))
                
                st.markdown("<h3 style='font-size: 32px;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
                results_df = pd.DataFrame(results_list)
                st.dataframe(results_df, use_container_width=True, height=400)
    
    with tab3:
        st.markdown("<h3 style='font-size: 28px;'>Upload CSV File</h3>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"<p style='font-size: 20px !important; color: #00d4ff !important;'><b>✅ Loaded {len(df):,} rows</b></p>", unsafe_allow_html=True)
            
            text_col = st.selectbox("Select text column:", df.columns.tolist())
            n_samples = st.slider("Number of samples to analyze", 5, min(50, len(df)), 10)
            
            if st.button("🔍 Analyze CSV", type="primary", key='analyze_csv'):
                sample_df = df.sample(n=n_samples)
                progress_bar = st.progress(0)
                results_list = []
                
                for idx, row in enumerate(sample_df.iterrows()):
                    text = str(row[1][text_col])
                    results = analyze_text(text, models)
                    results_list.append({
                        'Tweet': text[:100] + '...' if len(text) > 100 else text,
                        'Sarcasm': results['sarcasm']['label'],
                        'Confidence': f"{results['sarcasm']['confidence']:.1%}",
                        'Top Emotion': f"{get_emotion_emoji(results['emotions'][0]['label'])} {results['emotions'][0]['label']}",
                        'Emotion Score': f"{results['emotions'][0]['score']:.1%}",
                        'VADER': f"{results['vader']['compound']:.3f}"
                    })
                    progress_bar.progress((idx + 1) / n_samples)
                
                st.markdown("<h3 style='font-size: 32px;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
                results_df = pd.DataFrame(results_list)
                st.dataframe(results_df, use_container_width=True, height=400)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "analysis_results.csv",
                    "text/csv",
                    key='download-csv'
                )
