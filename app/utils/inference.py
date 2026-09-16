"""
Inference Module - Single and batch predictions
"""
import pandas as pd
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


def predict_single(
    text: str,
    model_type: str,
    model_loader
) -> Tuple[str, float, pd.Series]:
    """
    Predict emotion for a single text.
    
    Args:
        text: Customer message
        model_type: \"embedding_based\", \"lstm\", or \"gru\"
        model_loader: ModelLoader instance
    
    Returns:
        (emotion_label, confidence, probabilities_series)
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    # Predict
    emotion_label, confidence, probs_dict = model_loader.predict(text, model_type)
    
    # Convert to Series
    probs_series = pd.Series(probs_dict)
    
    return emotion_label, confidence, probs_series


def predict_batch(
    df: pd.DataFrame,
    text_column: str,
    model_type: str,
    model_loader
) -> pd.DataFrame:
    """
    Predict emotions for multiple texts.
    
    Args:
        df: DataFrame with messages
        text_column: Column name with text
        model_type: \"embedding_based\", \"lstm\", or \"gru\"
        model_loader: ModelLoader instance
    
    Returns:
        DataFrame with predictions
    """
    results = []
    
    for idx, row in df.iterrows():
        text = str(row[text_column])
        
        try:
            emotion, confidence, probs = predict_single(text, model_type, model_loader)
            
            result = {
                'original_text': text,
                'predicted_emotion': emotion,
                'confidence': confidence
            }
            
            # Add individual probabilities
            for emotion_class, prob in probs.items():
                result[f'prob_{emotion_class}'] = prob
            
            results.append(result)
        
        except Exception as e:
            results.append({
                'original_text': text,
                'predicted_emotion': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            })
    
    return pd.DataFrame(results)
