"""
Customer Sentiment & Emotion Intelligence Platform
Main Streamlit Application with Model Loading
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="Emotion Intelligence Platform",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.1em;
        color: #666;
    }
    .emotion-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/emotions.png", width=100)
    st.markdown("# 🧠 Emotion Platform")
    st.markdown("---")
    
    page = st.radio(
        "Navigate to:",
        ["🏠 Home", "💬 Single Prediction", "📊 Batch Inference", "📈 Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **Emotion Intelligence Platform** classifies customer messages into 6 emotions:
    - 😢 Sadness
    - 😊 Joy
    - 💕 Love
    - 😠 Anger
    - 😨 Fear
    - 😲 Surprise
    """)
    
    st.markdown("---")
    st.markdown("### Status")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.models_loaded:
            st.success("✓ Models Loaded")
        else:
            st.warning("⚠ No Models")

# Load models function
@st.cache_resource
def load_models():
    """Load models with caching"""
    try:
        from app.utils.model_loader import ModelLoader
        loader = ModelLoader(model_dir="models")
        return loader
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<div class="main-header">Customer Sentiment & Emotion Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify emotions, discover themes, and generate support intelligence</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Emotions", "6", "Classes")
    with col2:
        st.metric("Models", "2+", "Deep Learning")
    with col3:
        st.metric("Dataset", "16K+", "Training samples")
    
    st.markdown("---")
    
    # Load and check models
    loader = load_models()
    if loader:
        st.session_state.model_loader = loader
        st.session_state.models_loaded = True
        
        available = loader.list_available_models()
        
        st.success("### ✓ Models Ready for Inference")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Embedding-Based MLP**: {'✓ Ready' if available['embedding_based'] else '✗ Not Found'}")
        with col2:
            st.info(f"**LSTM**: {'✓ Ready' if available['lstm'] else '✗ Not Found'}")
        with col3:
            st.info(f"**GRU**: {'✓ Ready' if available['gru'] else '✗ Not Found'}")
    else:
        st.error("❌ Models not loaded. See below for setup instructions.")
    
    st.markdown("---")
    st.subheader("📌 Quick Start")
    st.markdown("""
    1. **Single Prediction** → Paste customer text and get emotion classification
    2. **Batch Inference** → Upload CSV with customer messages for bulk processing
    3. **Analytics** → View emotion distribution and insights
    """)
    
    st.subheader("🎯 Emotion Classes")
    emotions = {
        "😊 Joy": "Positive, happy, cheerful",
        "💕 Love": "Affection, care, warmth",
        "😢 Sadness": "Unhappy, sorrowful, disappointed",
        "😠 Anger": "Frustrated, annoyed, hostile",
        "😨 Fear": "Anxious, worried, scared",
        "😲 Surprise": "Astonished, amazed, shocked"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        for emotion, desc in list(emotions.items())[:3]:
            st.markdown(f"**{emotion}**: {desc}")
    with col2:
        for emotion, desc in list(emotions.items())[3:]:
            st.markdown(f"**{emotion}**: {desc}")
    
    st.markdown("---")
    st.subheader("⚙️ Setup Instructions")
    
    st.markdown("""
    ### Step 1: Export Models from Notebook
    Add this code at the end of `notebooks/MawddaNotebooks/01_Deep_Learning_Baseline.ipynb`:
    
    ```python
    # Export models
    embedding_model_path = "../../models/embedding_model.h5"
    lstm_model_path = "../../models/lstm_model.h5"
    
    model.save(embedding_model_path)
    lstm_model.save(lstm_model_path)
    
    print(f"✓ Embedding model saved to {embedding_model_path}")
    print(f"✓ LSTM model saved to {lstm_model_path}")
    ```
    
    ### Step 2: Verify Files Exist
    Check that these files are in `models/` directory:
    - ✓ `embedding_model.h5`
    - ✓ `lstm_model.h5`
    
    ### Step 3: Run Streamlit App
    ```bash
    streamlit run app/app.py
    ```
    """)

# SINGLE PREDICTION PAGE
elif page == "💬 Single Prediction":
    st.markdown("# 💬 Single Message Classification")
    st.markdown("Paste a customer message to classify its emotion.")
    
    loader = load_models()
    
    if loader is None:
        st.error("❌ Models not loaded. Please set up models first (see Home page)")
    else:
        available = loader.list_available_models()
        
        if not any(available.values()):
            st.error("❌ No models found in models/ directory")
            st.info("""
            **Expected files:**
            - models/embedding_model.h5
            - models/lstm_model.h5
            """)
        else:
            # Input
            user_text = st.text_area(
                "Enter customer message:",
                placeholder="e.g., 'I'm so frustrated with your customer service! This is the worst experience I've had.'",
                height=120
            )
            
            # Model selection
            model_options = []
            if available['embedding_based']:
                model_options.append("Embedding-Based MLP")
            if available['lstm']:
                model_options.append("LSTM")
            if available['gru']:
                model_options.append("GRU")
            
            if model_options:
                selected_model = st.radio(
                    "Select model:",
                    model_options,
                    horizontal=True
                )
                
                if st.button("🔍 Classify", use_container_width=True, key="single_predict"):
                    if not user_text.strip():
                        st.warning("Please enter a message to classify.")
                    else:
                        try:
                            with st.spinner("Processing..."):
                                from app.utils.inference import predict_single
                                
                                # Map display name to model type
                                model_type_map = {
                                    "Embedding-Based MLP": "embedding_based",
                                    "LSTM": "lstm",
                                    "GRU": "gru"
                                }
                                model_type = model_type_map[selected_model]
                                
                                emotion, confidence, probabilities = predict_single(
                                    user_text,
                                    model_type,
                                    loader
                                )
                            
                            # Results
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.success(f"### Predicted Emotion: **{emotion}**")
                                st.metric("Confidence", f"{confidence:.2%}")
                            
                            with col2:
                                st.write("### Probabilities")
                                st.bar_chart(probabilities)
                            
                            # Confidence interpretation
                            st.markdown("---")
                            if confidence >= 0.8:
                                st.info("✅ High confidence prediction")
                            elif confidence >= 0.6:
                                st.warning("⚠️ Moderate confidence - may need review")
                            else:
                                st.error("❌ Low confidence - ambiguous emotion")
                        
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                            st.info(f"Debug: {str(e)}")
            else:
                st.warning("No models available for prediction")

# BATCH INFERENCE PAGE
elif page == "📊 Batch Inference":
    st.markdown("# 📊 Batch CSV Classification")
    st.markdown("Upload a CSV file with customer messages for bulk emotion classification.")
    
    loader = load_models()
    
    if loader is None:
        st.error("❌ Models not loaded")
    else:
        available = loader.list_available_models()
        
        if not any(available.values()):
            st.error("❌ No models found")
        else:
            # File upload
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                st.info(f"📄 Loaded {len(df)} rows")
                st.dataframe(df.head(3), use_container_width=True)
                
                # Configuration
                st.subheader("⚙️ Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    text_column = st.selectbox("Select text column:", df.columns)
                
                with col2:
                    # Model selection
                    model_options = []
                    if available['embedding_based']:
                        model_options.append("Embedding-Based MLP")
                    if available['lstm']:
                        model_options.append("LSTM")
                    if available['gru']:
                        model_options.append("GRU")
                    
                    selected_model = st.selectbox("Select model:", model_options)
                
                if st.button("🚀 Process All Messages", use_container_width=True, key="batch_predict"):
                    try:
                        with st.spinner(f"Processing {len(df)} messages..."):
                            from app.utils.inference import predict_batch
                            
                            # Map display name to model type
                            model_type_map = {
                                "Embedding-Based MLP": "embedding_based",
                                "LSTM": "lstm",
                                "GRU": "gru"
                            }
                            model_type = model_type_map[selected_model]
                            
                            results_df = predict_batch(
                                df,
                                text_column,
                                model_type,
                                loader
                            )
                        
                        st.success(f"✅ Processed {len(results_df)} messages")
                        
                        # Results
                        st.subheader("📋 Results")
                        st.dataframe(results_df[['original_text', 'predicted_emotion', 'confidence']], use_container_width=True)
                        
                        # Summary
                        st.subheader("📈 Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_conf = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf:.2%}")
                        
                        with col2:
                            low_conf = (results_df['confidence'] < 0.6).sum()
                            st.metric("Low Confidence (<60%)", low_conf)
                        
                        with col3:
                            errors = (results_df['predicted_emotion'] == 'ERROR').sum()
                            st.metric("Processing Errors", errors)
                        
                        # Emotion distribution
                        st.subheader("😊 Emotion Distribution")
                        emotion_counts = results_df['predicted_emotion'].value_counts()
                        st.bar_chart(emotion_counts)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="emotion_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        st.info(f"Debug info: {str(e)}")

# ANALYTICS PAGE
elif page == "📈 Analytics":
    st.markdown("# 📈 Analytics & Trends")
    
    loader = load_models()
    
    if loader:
        st.info("""
        📌 **Upload sample predictions** in the Batch Inference page to see analytics here.
        
        Planned features:
        - Emotion distribution trends
        - Confidence score analysis
        - Model performance metrics
        - Per-emotion statistics
        """)
    else:
        st.error("Models not loaded")

st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.8em; color: #999'>
Emotion Intelligence Platform | TechTrek Advanced Data Science & AI
</div>
""", unsafe_allow_html=True)
