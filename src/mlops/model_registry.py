"""
MLflow Model Registry
Track and version emotion classification models
"""
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class EmotionModelRegistry:
    """
    Registry for tracking trained emotion models using MLflow.
    """
    
    def __init__(self, registry_dir: str = "mlruns"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.emotion_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file:{self.registry_dir.absolute()}")
    
    def log_model(
        self,
        model,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        description: str = ""
    ):
        """
        Log a trained model with metrics and parameters.
        
        Example:
        ```python
        registry = EmotionModelRegistry()
        registry.log_model(
            model=model,
            model_name="embedding_baseline",
            model_type="embedding_based",
            metrics={'accuracy': 0.79, 'macro_f1': 0.65},
            params={'epochs': 10, 'batch_size': 32},
            description="Embedding-based MLP baseline"
        )
        ```
        """
        
        if not MLFLOW_AVAILABLE:
            print("⚠ MLflow not installed. Logging locally instead.")
            self._log_model_local(model_name, model_type, metrics, params, description)
            return
        
        try:
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log tags
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("emotion_classes", ",".join(self.emotion_classes))
                mlflow.set_tag("description", description)
                
                # Log model
                mlflow.keras.log_model(model, artifact_path="model")
                
                print(f"✓ Logged {model_name} to MLflow")
        
        except Exception as e:
            print(f"Error logging to MLflow: {e}. Using local logging.")
            self._log_model_local(model_name, model_type, metrics, params, description)
    
    def _log_model_local(
        self,
        model_name: str,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        description: str
    ):
        """Log model metadata locally (fallback)"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_type': model_type,
            'description': description,
            'metrics': metrics,
            'params': params,
            'emotion_classes': self.emotion_classes
        }
        
        metadata_path = self.registry_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Logged {model_name} metadata to {metadata_path}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a logged model"""
        metadata_path = self.registry_dir / f"{model_name}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def list_models(self):
        """List all logged models"""
        models = []
        for metadata_file in self.registry_dir.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                models.append(metadata)
        return models


def log_training_run(
    model,
    model_name: str,
    model_type: str,
    test_metrics: Dict[str, float],
    hyperparams: Dict[str, Any],
    description: str = ""
):
    """
    Convenience function to log a training run.
    
    Example in notebook:
    ```python
    from src.mlops.model_registry import log_training_run
    
    log_training_run(
        model=model,
        model_name="embedding_baseline",
        model_type="embedding_based",
        test_metrics={'accuracy': 0.79, 'macro_f1': 0.65},
        hyperparams={'epochs': 10, 'batch_size': 32},
        description="Embedding-based MLP baseline"
    )
    ```
    """
    registry = EmotionModelRegistry()
    registry.log_model(
        model=model,
        model_name=model_name,
        model_type=model_type,
        metrics=test_metrics,
        params=hyperparams,
        description=description
    )


if __name__ == "__main__":
    registry = EmotionModelRegistry()
    print("Registry initialized at:", registry.registry_dir)