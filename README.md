# Sentiment_Analysis_XAI_Sarcasm
SentiSarc XAI Platform is a cutting-edge web application that combines state-of-the-art Natural Language Processing with Explainable Artificial Intelligence (XAI) to detect sarcasm, analyze emotions, and provide transparent insights into AI decision-making processes.
Unlike traditional sentiment analysis tools, SentiSarc specializes in understanding the complex nuances of sarcastic language while providing complete transparency into how the AI reaches its conclusions through advanced XAI techniques.

Key Features
🤖 Multi-Model AI Analysis
Sarcasm Detection: Powered by RoBERTa transformer model fine-tuned on irony detection

28-Emotion Classification: Comprehensive emotion analysis using Google's GoEmotions taxonomy

Sentiment Analysis: VADER sentiment scoring for additional context

Real-time Processing: Instant analysis with live confidence scoring

🔍 Explainable AI (XAI)
LIME Explanations: Local Interpretable Model-agnostic Explanations showing feature importance
SHAP Values: Game-theoretic approach to feature contribution analysis
Interactive Visualizations: Beautiful, interpretable charts showing model reasoning
Token-level Insights: Understand which words drive sarcasm predictions

📊 Multiple Analysis Modes
💬 Single Tweet Analysis: Deep-dive analysis of individual text inputs

📊 Batch Processing: Analyze multiple tweets simultaneously with summary statistics

🎲 Random Sampling: Explore datasets with intelligent random tweet generation

💾 CSV Upload: Process custom datasets with flexible column mapping

🎨 User Experience
Futuristic UI: Cyberpunk-inspired interface with gradient animations
Responsive Design: Optimized for both desktop and mobile devices
Interactive Charts: Plotly-powered visualizations with hover effects
Real-time Updates: Live progress tracking and instant results

🤖 AI Models & Technologies
Core AI Models
Sarcasm Detection
Model: cardiffnlp/twitter-roberta-base-irony
Architecture: RoBERTa-base transformer (125M parameters)
Training Data: Twitter irony detection dataset from Cardiff NLP
Task: Binary classification (Sarcastic vs Not Sarcastic)
Input: 512 token maximum length
Output: Probability scores for both classes

Emotion Classification
Model: SamLowe/roberta-base-go_emotions
Architecture: RoBERTa-base transformer (125M parameters)
Training Data: GoEmotions dataset (58k Reddit comments)
Task: Multi-label classification across 28 emotions
Emotion Categories: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
Output: Confidence scores for all 28 emotions

Sentiment Analysis
Model: VADER (Valence Aware Dictionary and sEntiment Reasoner)
Type: Rule-based lexicon and sentiment analysis tool
Specialization: Social media text and informal language
Output: Compound score (-1.0 to +1.0) + individual sentiment components

Explainable AI (XAI) Frameworks
LIME (Local Interpretable Model-agnostic Explanations)
Purpose: Local surrogate model for feature importance
Method: Perturbs input data and observes prediction changes
Configuration: 100 samples, 8 top features, text-based explanations
Output: Word-level importance scores for predictions

SHAP (SHapley Additive exPlanations)
Purpose: Game-theoretic feature attribution
Method: Computes Shapley values by feature ablation
Implementation: Custom approximation for text classification
Output: Contribution scores for each word/token

Technical Stack
Backend & Framework
Web Framework: Streamlit
Language: Python 3.8+
ML Framework: PyTorch
NLP Library: Hugging Face Transformers
Visualization: Plotly Graph Objects

Libraries & Dependencies
Data Processing: pandas, numpy
NLP Tools: transformers, tokenizers
Sentiment Analysis: vaderSentiment
XAI: lime, shap
Utilities: warnings, os, random, pickle
