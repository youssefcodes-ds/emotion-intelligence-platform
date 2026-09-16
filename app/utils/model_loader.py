"""
Model Loader - Load trained TensorFlow/Keras models
"""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import warnings

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None


class ModelLoader:
    """
    Loads trained emotion classification models from disk.
    Supports TensorFlow/Keras .h5 format models.
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize model loader"""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.loaded_successfully = False
        
        # Emotion mapping
        self.emotion_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.emotion_emoji = {
            'sadness': '😢',
            'joy': '😊',
            'love': '💕',
            'anger': '😠',
            'fear': '😨',
            'surprise': '😲'
        }
        
        # Load models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models from disk"""
        
        if not self.model_dir.exists():
            print(f"⚠ Model directory not found: {self.model_dir}")
            return
        
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not installed!")
            return
        
        # Load Embedding-based model
        embedding_path = self.model_dir / "embedding_model.h5"
        if embedding_path.exists():
            try:
                self.models['embedding_based'] = tf.keras.models.load_model(str(embedding_path))
                print(f"✓ Loaded Embedding-Based MLP: {embedding_path}")
            except Exception as e:
                print(f"✗ Failed to load Embedding model: {e}")
        else:
            print(f"⚠ Embedding model not found: {embedding_path}")
        
        # Load LSTM model
        lstm_path = self.model_dir / "lstm_model.h5"
        if lstm_path.exists():
            try:
                self.models['lstm'] = tf.keras.models.load_model(str(lstm_path))
                print(f"✓ Loaded LSTM: {lstm_path}")
            except Exception as e:
                print(f"✗ Failed to load LSTM model: {e}")
        else:
            print(f"⚠ LSTM model not found: {lstm_path}")
        
        # Load GRU model (optional)
        gru_path = self.model_dir / "gru_model.h5"
        if gru_path.exists():
            try:
                self.models['gru'] = tf.keras.models.load_model(str(gru_path))
                print(f"✓ Loaded GRU: {gru_path}")
            except Exception as e:
                print(f"✗ Failed to load GRU model: {e}")
        
        if self.models:
            self.loaded_successfully = True
            print(f"\n✓ Successfully loaded {len(self.models)} model(s)")
        else:
            print("\n❌ No models loaded successfully")
    
    def list_available_models(self) -> Dict[str, bool]:
        """Return dict of available models"""
        return {
            'embedding_based': 'embedding_based' in self.models,
            'lstm': 'lstm' in self.models,
            'gru': 'gru' in self.models,
        }
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """Get model by type"""
        return self.models.get(model_type)
    
    def predict(self, text: str, model_type: str = "embedding_based") -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion for text.
        
        Returns:
            (emotion_label, confidence, probabilities_dict)
        """
        model = self.get_model(model_type)
        if model is None:
            raise ValueError(f"Model '{model_type}' not loaded")
        
        # Predict
        try:
            predictions = model.predict(np.array([text]), verbose=0)
            probs = predictions[0]
            class_idx = np.argmax(probs)
            confidence = float(probs[class_idx])
            
            # Create probability dict
            probs_dict = {
                self.emotion_classes[i]: float(probs[i])
                for i in range(len(self.emotion_classes))
            }
            
            emotion_label = self.get_emotion_label(class_idx)
            
            return emotion_label, confidence, probs_dict
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_emotion_label(self, class_idx: int, with_emoji: bool = True) -> str:
        """Convert class index to emotion label"""
        if 0 <= class_idx < len(self.emotion_classes):
            emotion = self.emotion_classes[class_idx]
            if with_emoji:
                emoji = self.emotion_emoji.get(emotion, '')
                return f"{emoji} {emotion.capitalize()}"
            return emotion.capitalize()
        return "Unknown"
    
    def get_emotion_index(self, emotion: str) -> Optional[int]:
        """Convert emotion name to class index"""
        emotion_lower = emotion.lower().strip()
        if emotion_lower in self.emotion_classes:
            return self.emotion_classes.index(emotion_lower)
        return None


if __name__ == "__main__":
    loader = ModelLoader()
    print("\nAvailable models:", loader.list_available_models())
    print("Emotions:", [loader.get_emotion_label(i) for i in range(6)])
