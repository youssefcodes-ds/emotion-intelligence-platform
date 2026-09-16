"""
Visualization Module
Create charts and analytics visualizations for the Streamlit app.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional


def emotion_distribution_chart(emotion_counts: pd.Series, title: str = "Emotion Distribution"):
    """Create bar chart for emotion distribution."""
    
    emotion_emoji = {
        'sadness': '😢',
        'joy': '😊',
        'love': '💕',
        'anger': '😠',
        'fear': '😨',
        'surprise': '😲'
    }
    
    # Format labels with emoji
    labels = [f"{emotion_emoji.get(e.lower().split()[0], '')} {e}" for e in emotion_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=emotion_counts.values,
            marker=dict(
                color=emotion_counts.values,
                colorscale='Viridis'
            ),
            text=emotion_counts.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Emotion",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    
    return fig


def confidence_distribution_chart(confidences: List[float], title: str = "Prediction Confidence Distribution"):
    """Create histogram of prediction confidences."""
    
    fig = go.Figure(data=[
        go.Histogram(
            x=confidences,
            nbinsx=30,
            marker=dict(color='rgba(31, 119, 180, 0.7)'),
            name='Confidence'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    return fig


def confusion_matrix_heatmap(confusion_matrix: np.ndarray, emotion_classes: List[str], title: str = "Confusion Matrix"):
    """Create heatmap for confusion matrix."""
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=emotion_classes,
        y=emotion_classes,
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Emotion",
        yaxis_title="True Emotion",
        height=600,
        width=700
    )
    
    return fig


def per_class_metrics_chart(metrics_df: pd.DataFrame, title: str = "Per-Class Metrics"):
    """Create bar chart comparing per-class metrics (precision, recall, F1)."""
    
    fig = go.Figure()
    
    for metric in ['precision', 'recall', 'f1-score']:
        if metric in metrics_df.columns:
            fig.add_trace(go.Bar(
                x=metrics_df.index,
                y=metrics_df[metric],
                name=metric.capitalize()
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Emotion Class",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig


def model_comparison_chart(model_metrics: Dict[str, Dict[str, float]], metric_name: str = "F1 Score"):
    """Compare metrics across models."""
    
    fig = go.Figure()
    
    for model_name, metrics in model_metrics.items():
        fig.add_trace(go.Bar(
            name=model_name,
            x=list(metrics.keys()),
            y=list(metrics.values())
        ))
    
    fig.update_layout(
        title=f"{metric_name} by Model and Emotion",
        xaxis_title="Emotion",
        yaxis_title=metric_name,
        barmode='group',
        height=400
    )
    
    return fig


def probability_radar_chart(probabilities: Dict[str, float], title: str = "Emotion Probabilities"):
    """Create radar chart for emotion probabilities."""
    
    emotions = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Add first emotion at end to close the circle
    emotions.append(emotions[0])
    values.append(values[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='rgba(31, 119, 180, 0.8)')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        height=500,
        showlegend=False
    )
    
    return fig


def emotion_timeline_chart(df: pd.DataFrame, date_column: str = "date", emotion_column: str = "emotion", title: str = "Emotion Trend Over Time"):
    """Create line chart showing emotion trends over time."""
    
    if date_column not in df.columns or emotion_column not in df.columns:
        return None
    
    # Group by date and emotion
    timeline_df = df.groupby([date_column, emotion_column]).size().reset_index(name='count')
    
    fig = px.line(
        timeline_df,
        x=date_column,
        y='count',
        color=emotion_column,
        title=title,
        labels={'count': 'Number of Messages', date_column: 'Date', emotion_column: 'Emotion'},
        height=400
    )
    
    return fig


def create_summary_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """Create summary statistics from predictions."""
    
    metrics = {
        'total_messages': len(df),
        'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
        'low_confidence_count': (df['confidence'] < 0.6).sum() if 'confidence' in df.columns else 0,
        'emotion_distribution': df['predicted_emotion'].value_counts() if 'predicted_emotion' in df.columns else None
    }
    
    return metrics


if __name__ == "__main__":
    # Test visualizations
    import pandas as pd
    
    # Sample data
    emotions = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
    counts = pd.Series([120, 45, 60, 35, 90, 50], index=emotions)
    
    # Test emotion distribution
    fig1 = emotion_distribution_chart(counts)
    print("✓ Emotion distribution chart created")
    
    # Test confidence distribution
    confidences = np.random.uniform(0.5, 1.0, 100).tolist()
    fig2 = confidence_distribution_chart(confidences)
    print("✓ Confidence distribution chart created")
    
    # Test probability radar
    probs = {e: np.random.uniform(0, 1) for e in emotions}
    probs = {k: v / sum(probs.values()) for k, v in probs.items()}
    fig3 = probability_radar_chart(probs)
    print("✓ Probability radar chart created")
    
    print("\nAll visualizations tested successfully!")
