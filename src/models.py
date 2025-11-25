import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lime.lime_text import LimeTextExplainer
import streamlit as st

@st.cache_resource
def load_models():
    """Load all models with caching"""
    try:
        models = {}
        
        sarcasm_model_name = "cardiffnlp/twitter-roberta-base-irony"
        st.info(f"🔄 Loading sarcasm model ({sarcasm_model_name}) from HuggingFace...")
        models['sarcasm_tokenizer'] = AutoTokenizer.from_pretrained(sarcasm_model_name)
        models['sarcasm_model'] = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_name)
        
        st.info("🔄 Loading emotion classifier...")
        models['emotion_classifier'] = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            device=0 if torch.cuda.is_available() else -1
        )
        
        models['vader'] = SentimentIntensityAnalyzer()
        
        models['lime_explainer'] = LimeTextExplainer(
            class_names=['Not Sarcastic', 'Sarcastic'],
            bow=False
        )
        
        st.success("✅ All models loaded successfully!")
        return models
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

def predict_sarcasm(text, tokenizer, model):
    """Predict sarcasm with confidence scores"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return {
            'label': 'Sarcastic' if prediction == 1 else 'Not Sarcastic',
            'confidence': confidence,
            'prob_not_sarcastic': probs[0][0].item(),
            'prob_sarcastic': probs[0][1].item()
        }
    except Exception as e:
        st.error(f"Sarcasm prediction error: {e}")
        return {
            'label': 'Error',
            'confidence': 0.0,
            'prob_not_sarcastic': 0.5,
            'prob_sarcastic': 0.5
        }

def predict_emotion(text, classifier):
    """Predict emotions with full GoEmotions label set"""
    try:
        results = classifier(text)[0]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results
    except Exception as e:
        st.error(f"Emotion prediction error: {e}")
        return [{'label': 'neutral', 'score': 1.0}]

def get_vader_sentiment(text, vader):
    """Get VADER sentiment scores"""
    scores = vader.polarity_scores(text)
    return scores

def analyze_text(text, models):
    """Complete text analysis"""
    sarcasm_result = predict_sarcasm(text, models['sarcasm_tokenizer'], models['sarcasm_model'])
    emotion_results = predict_emotion(text, models['emotion_classifier'])
    vader_scores = get_vader_sentiment(text, models['vader'])
    
    return {
        'sarcasm': sarcasm_result,
        'emotions': emotion_results,
        'vader': vader_scores
    }
