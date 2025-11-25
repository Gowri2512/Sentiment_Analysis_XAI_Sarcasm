import numpy as np
import plotly.graph_objects as go
import pandas as pd
from .models import predict_sarcasm

def explain_with_lime(text, models):
    """Generate LIME explanation with optimized performance"""
    def predictor(texts):
        results = []
        for txt in texts:
            pred = predict_sarcasm(txt, models['sarcasm_tokenizer'], models['sarcasm_model'])
            results.append([pred['prob_not_sarcastic'], pred['prob_sarcastic']])
        return np.array(results)
    
    exp = models['lime_explainer'].explain_instance(
        text,
        predictor,
        num_features=8,
        num_samples=100,
        top_labels=1
    )
    return exp

def explain_with_shap(text, models):
    """Generate SHAP explanation with optimized performance"""
    try:
        words = text.split()
        base_pred = predict_sarcasm(text, models['sarcasm_tokenizer'], models['sarcasm_model'])
        base_score = base_pred['prob_sarcastic']
        
        word_scores = []
        for i in range(len(words)):
            temp_words = words[:i] + words[i+1:]
            temp_text = ' '.join(temp_words)
            
            if temp_text.strip():
                temp_pred = predict_sarcasm(temp_text, models['sarcasm_tokenizer'], models['sarcasm_model'])
                importance = base_score - temp_pred['prob_sarcastic']
                word_scores.append(importance)
            else:
                word_scores.append(0.0)
        
        return {
            'words': words,
            'scores': word_scores,
            'base_value': 0.5,
            'final_value': base_score
        }
        
    except Exception as e:
        print(f"⚠️ SHAP explanation not available: {str(e)}")
        return None
