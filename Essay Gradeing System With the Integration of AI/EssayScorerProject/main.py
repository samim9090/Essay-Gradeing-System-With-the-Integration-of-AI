import streamlit as st
import pickle
import os
from utils.processor import extract_features
from utils.feedback import get_feedback

# Set page configuration
st.set_page_config(page_title="AI Essay Scorer", page_icon="📝", layout="wide")

# --- Sidebar: App Info ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("**Model:** `llama-3.3-70b-versatile`")
    st.markdown("**Scorer:** TF-IDF + RandomForest")
    st.divider()
    st.markdown("Paste your essay, click **Score Essay**, and get an instant score plus detailed AI feedback.")

# Hardcoded Groq API Key
GROQ_API_KEY = "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"

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'essay_scorer_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model not found at {model_path}. Please ensure the model file exists.")
        return None

def main():
    st.title("📝 AI Essay Scorer & Feedback System")
    st.markdown("""
    This application scores your essay using a trained AI model and provides detailed feedback 
    powered by **Groq** ⚡ and **LLaMA 3.3-70b**.
    """)

    # Use hardcoded API Key
    API_KEY = GROQ_API_KEY

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        essay_text = st.text_area("✍️ Paste your essay here:", height=400, placeholder="Write or paste your essay content here...")
        
    with col2:
        st.info("How it works:")
        st.markdown("""
        1. **Paste your essay** in the text area.
        2. **Click 'Score Essay'** to get an instant score.
        3. **Read the feedback** to improve your writing.
        """)
        
        if st.button("Score Essay", type="primary", use_container_width=True):
            if essay_text and len(essay_text.strip()) > 10:
                with st.spinner("Analyzing essay..."):
                    # 1. Load Model
                    model = load_model()
                    
                    if model:
                        # 2. Score Essay
                        # The model expects raw text if it's a pipeline with vectorizer
                        # otherwise we might need to extract features first.
                        # Based on training script: model = make_pipeline(TfidfVectorizer(), RandomForestRegressor())
                        # So we pass raw text directly.
                        try:
                            score = model.predict([essay_text])[0]
                            st.success(f"### Predicted Score: {score:.2f} / 10")
                            st.progress(min(score/10, 1.0))
                        except Exception as e:
                            st.error(f"Error scoring essay: {e}")
                            score = 0

                        # 3. Get Feedback
                        if API_KEY:
                            st.divider()
                            st.subheader("🤖 AI Feedback  *(Groq · LLaMA 3.3-70b)*")
                            with st.spinner("Generating feedback via Groq..."):
                                feedback = get_feedback(essay_text, score, API_KEY)
                                st.markdown(feedback)
                        else:
                            st.warning("⬅️ Enter your Groq API key in the sidebar to get AI feedback.")
            else:
                st.warning("Please enter a valid essay (at least 10 characters).")

    # Metrics section
    if essay_text:
        st.divider()
        st.subheader("📊 Text Metrics")
        features = extract_features(essay_text)
        m1, m2, m3 = st.columns(3)
        m1.metric("Word Count", features["word_count"])
        m2.metric("Sentence Count", features["sentence_count"])
        m3.metric("Avg Word Length", f"{features['avg_word_length']:.2f}")

if __name__ == "__main__":
    main()
