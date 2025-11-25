import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from .utils import get_emotion_emoji

def create_emotion_chart(emotions):
    """Create interactive emotion bar chart"""
    df = pd.DataFrame(emotions[:10])
    df['emoji'] = df['label'].apply(get_emotion_emoji)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['score'],
            y=[f"{row['emoji']} {row['label']}" for _, row in df.iterrows()],
            orientation='h',
            marker=dict(
                color=df['score'],
                colorscale='Plasma',
                line=dict(color='rgba(0,212,255,0.8)', width=2)
            ),
            text=[f"{score:.2%}" for score in df['score']],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Orbitron')
        )
    ])
    
    fig.update_layout(
        title="Emotion Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Emotion",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Rajdhani', size=14),
        height=450,
        showlegend=False
    )
    
    return fig

def create_sarcasm_gauge(confidence, label):
    """Create sarcasm confidence gauge"""
    color = '#f72585' if label == 'Sarcastic' else '#00d4ff'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{label}</b>",
               'font': {'size': 24, 'color': '#ffffff', 'family': 'Orbitron'}},
        number={'font': {'size': 48, 'color': color, 'family': 'Orbitron'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "#ffffff", 'tickfont': {'size': 14}},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 3,
            'bordercolor': color,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0,212,255,0.2)'},
                {'range': [50, 100], 'color': 'rgba(247,37,133,0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.9,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff", 'family': 'Orbitron'},
        height=350
    )
    
    return fig

def create_vader_chart(vader_scores):
    """Create VADER sentiment visualization"""
    labels = ['Negative', 'Neutral', 'Positive']
    values = [vader_scores['neg'], vader_scores['neu'], vader_scores['pos']]
    colors = ['#f72585', '#7b2ff7', '#00d4ff']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2),
                pattern=dict(shape="/", solidity=0.3)
            ),
            text=[f"{v:.2%}" for v in values],
            textposition='auto',
            textfont=dict(size=16, color='white', family='Orbitron')
        )
    ])
    
    fig.update_layout(
        title=f"<b>VADER Sentiment (Compound: {vader_scores['compound']:.3f})</b>",
        yaxis_title="Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Rajdhani', size=14),
        height=350,
        showlegend=False
    )
    
    return fig
