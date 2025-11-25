import pandas as pd
import os
import random

def get_emotion_emoji(emotion):
    """Map emotions to emojis"""
    emoji_map = {
        'admiration': '👏', 'amusement': '😄', 'anger': '😠', 'annoyance': '😒',
        'approval': '👍', 'caring': '🤗', 'confusion': '😕', 'curiosity': '🤔',
        'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
        'embarrassment': '😳', 'excitement': '🎉', 'fear': '😨', 'gratitude': '🙏',
        'grief': '😢', 'joy': '😊', 'love': '❤️', 'nervousness': '😰',
        'optimism': '🌟', 'pride': '🦁', 'realization': '💡', 'relief': '😌',
        'remorse': '😔', 'sadness': '😢', 'surprise': '😲', 'neutral': '😐'
    }
    return emoji_map.get(emotion.lower(), '🎭')

def load_dataset_tweets():
    """Load tweets from dataset with better error handling"""
    try:
        possible_paths = [
            'data/eng_dataset.csv',
            'data/preprocessed.csv',
            'eng_dataset.csv',
            'preprocessed.csv',
            '../data/eng_dataset.csv',
            '../data/preprocessed.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
        
        return None
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def generate_random_tweets(df, n=10):
    """Generate random tweets from dataset"""
    if df is not None and len(df) > 0:
        possible_cols = ['text', 'tweet', 'content', 'message', 'Text', 'Tweet', 'Content', 'Message']
        text_col = None
        
        for col in possible_cols:
            if col in df.columns:
                text_col = col
                break
        
        if text_col:
            tweets = df[text_col].dropna().astype(str).sample(min(n, len(df))).tolist()
            return tweets
    
    # Fallback sample tweets
    fallback_tweets = [
        "Just love it when my code works on the first try... said no one ever! 😂",
        "Oh great, another meeting that could have been an email.",
        "I'm so excited to do my taxes this weekend!",
        "Nothing better than waiting in traffic for hours!",
        "This weather is absolutely perfect for staying indoors all day.",
    ]
    return fallback_tweets[:n]
