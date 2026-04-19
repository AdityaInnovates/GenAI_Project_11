import streamlit as st
import joblib
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.figure_factory as ff

# Import the LangGraph agent workflow
import sys
sys.path.append(os.path.dirname(__file__))
from Agent.model import workflow
st.set_page_config(
    page_title="Intelligent News Credibility Analyzer",
    page_icon="📰",
    layout="centered"
)

# extract text from url
def extract_text(url):
    """Fetch and extract article text from a URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)
    return text.strip()




# loading the model and vectorizer . 
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)  # folder where app.py is
    model_path = os.path.join(base_dir, "ml", "model.pkl")
    vectorizer_path = os.path.join(base_dir, "ml", "vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model , vectorizer = load_model()

# loading the test data for model performance metrics
@st.cache_data
def load_test_data():
    base_dir = os.path.dirname(__file__)
    ml_dir = os.path.join(base_dir, "ml")
    X_test = pickle.load(open(os.path.join(ml_dir, "X_test.pkl"), "rb"))
    y_test = pickle.load(open(os.path.join(ml_dir, "y_test.pkl"), "rb"))
    return X_test, y_test

X_test, y_test = load_test_data()


# making the ui of the app . 
st.title("📰 Intelligent News Credibility Analyzer")
st.caption("ML-based News Credibility Analysis (No LLMs)")

st.markdown(
    """
This system analyzes **news articles** using **classical NLP & Machine Learning**
to assess **credibility risk** based on textual patterns.
"""
)

# -------- MODEL PERFORMANCE SECTION --------
with st.expander("📈 Model Performance Metrics", expanded=False):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Metric cards in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.markdown("---")

    # Confusion Matrix Heatmap
    st.subheader("Confusion Matrix")
    labels = ["Fake (0)", "Real (1)"]
    cm_text = [[str(val) for val in row] for row in cm]
    fig = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=labels,
        y=labels,
        annotation_text=cm_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        xaxis=dict(side="bottom"),
        width=500,
        height=400,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# taking input from the user .
input_type = st.radio(
    "Choose input method:",
    ["Paste Article Text", "Enter Article URL"]
)

# Initialize session state for article text
if "article_text" not in st.session_state:
    st.session_state.article_text = ""

article_text = ""

if input_type == "Paste Article Text":
    article_text = st.text_area(
        "Paste news article text here",
        height=250,
        placeholder="Paste the full article content..."
    )
    st.session_state.article_text = article_text

else:
    url = st.text_input("Enter news article URL")
    if st.button("Fetch Article"):
        if url.strip():
            with st.spinner("Extracting article..."):
                try:
                    st.session_state.article_text = extract_text(url)
                    st.success("Article extracted successfully!")
                    st.text_area("Extracted Article", st.session_state.article_text, height=250)
                except Exception as e:
                    st.error("Failed to extract article")
                    st.session_state.article_text = ""
    
    # Use the stored article text
    article_text = st.session_state.article_text

# anahlyzing the credibility of the article .
if st.button("Analyze Credibility"):

    if not article_text.strip():
        st.warning("Please provide article text or URL.")
    else:
        with st.spinner("Analyzing credibility..."):
            
            transformed = vectorizer.transform([article_text])
            prediction = model.predict(transformed)[0]
            probabilities = model.predict_proba(transformed)[0]

            confidence = np.max(probabilities)

        st.divider()

        # Result 
        if prediction == 1:
            st.success("✅ High Credibility Detected")
        else:
            st.error("⚠️ Low Credibility Detected")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}"
        )

        # Explanation 
        with st.expander("📊 How was this decision made?"):
            st.write(
                """
- Text was cleaned and vectorized using **TF-IDF**
- Model used: **Logistic Regression**
- Decision based on learned linguistic & credibility patterns
- No external APIs or LLMs were used
                """
            )

        # -------- DISCLAIMER --------
        with st.expander("⚖️ Ethical Disclaimer"):
            st.write(
                """
This tool provides **probabilistic analysis**, not absolute truth.
It should be used as a **decision-support system**, not a final authority.
                """
            )

        st.markdown("---")
        st.subheader("🕵️‍♂️ Agentic Fact-Checking (Tavily Search + Llama 3.1)")
        st.caption("Validates specific claims dynamically across the web using Tavily Search.")
        
        with st.spinner("🤖 Agent is extracting claims and checking facts... This may take a minute."):
            try:
                # Invoke the LangGraph workflow
                initial_state = {"article_text": article_text}
                final_state = workflow.invoke(initial_state)
                
                claims = final_state.get("extracted_claims", [])
                retrieval_results = final_state.get("retrieval_results", {})
                final_report = final_state.get("final_report", "No report generated.")
                
                st.success("Analysis Complete!")
                
                # Show extracted claims
                with st.expander("📋 Extracted Factual Claims", expanded=True):
                    if not claims:
                        st.info("No verifiable factual claims were extracted.")
                    else:
                        for c in claims:
                            st.write(f"- **{c.entity}**: {c.claim}")
                            
                # Show RAG Evidence
                with st.expander("🌐 Retrieved Evidence from Web Search", expanded=False):
                    for claim, evidence in retrieval_results.items():
                        st.markdown(f"**Claim:** {claim}")
                        if "NO VERIFIED EVIDENCE FOUND" in evidence:
                            st.warning(evidence)
                        else:
                            st.success("Evidence found in database!")
                            st.text(evidence[:500] + "..." if len(evidence) > 500 else evidence)
                        st.divider()
                
                # Show Final Report
                st.subheader("📝 Final Credibility Report")
                st.markdown(final_report)
                
            except Exception as e:
                st.error(f"Failed to run agentic analysis: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Academic Project | NLP & ML & Generative AI | Deployed via Streamlit")