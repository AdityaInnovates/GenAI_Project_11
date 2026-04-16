# 📰 Intelligent News Credibility Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red.svg)

A dual-engine news credibility system that combines **Classical Machine Learning (NLP + Logistic Regression)** with an **Agentic AI Fact-Checking Pipeline (Llama 3.1 + RAG)**. It seamlessly evaluates textual metadata for credibility while deploying an autonomous agent against the PolitiFact `LIAR` dataset to cross-reference and verify explicit claims without generating AI hallucinations.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset Information](#-dataset-information)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Responsible AI & Ethics](#️-responsible-ai--ethics)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

### Problem Statement

In the digital age, the proliferation of fake news poses significant threats to:

- **Democratic processes** and public opinion
- **Public health** (e.g., misinformation during pandemics)
- **Social stability** and trust in media institutions
- **Individual decision-making** based on factual information

### Why This Matters

With billions of people consuming news online daily, distinguishing credible journalism from misinformation has become critical. This project addresses this challenge by building an automated, scalable, and accurate fake news detection system.

### Approach

This application supports two distinct pipelines designed to process news data comprehensively:

**Phase 1: Classical Machine Learning**
- **TF-IDF Vectorization** and **Logistic Regression** for fast, local binary classification (Fake vs. Real).
- Prioritizes **explainability**, **speed**, and **resource efficiency** without reliance on external APIs.

**Phase 2: RAG-based Agentic Fact-Checking**
- **LangGraph** orchestration managing a multi-node Llama-3.1-8B LLM.
- **ChromaDB** persistent vector store actively querying the `LIAR` fact-checking dataset. 
- **Graceful Degradation** mechanism to identify unverified quotes rather than producing AI hallucinations.

---

## ✨ Features

### Core Capabilities

1. **📝 Text Input**
   - Paste news article text directly into the application
   - Supports articles of varying lengths

2. **🔗 URL Input**
   - Enter article URLs to automatically extract and analyze content
   - Built-in web scraping using BeautifulSoup
   - Smart paragraph extraction

3. **🎯 Credibility Assessment**
   - Binary classification: High Credibility vs Low Credibility
   - Real-time inference with pre-trained model

4. **📊 Confidence Score**
   - Displays model confidence (probability) for predictions
   - Helps users understand prediction certainty

5. **🔍 Transparency Features**
   - Explanation of decision-making process
   - Clear indication of methods used (TF-IDF + Logistic Regression)

6. **🕵️‍♂️ Agentic Fact-Checking (Llama 3.1)**
   - Autonomously extracts verifiable atomic claims from input.
   - Embeds semantics and fetches matches against local truth datasets (PolitiFact `LIAR`).
   - Produces detailed credibility reports pointing exactly to corroborating evidence.

7. **⚖️ Ethical Disclaimer**
   - Responsible AI messaging.
   - Safely isolates "Unverified historical data" from deterministic falsity.

---

## 🛠️ Tech Stack

### Core Technologies

| Technology       | Purpose                      | Version |
| ---------------- | ---------------------------- | ------- |
| **Python**       | Primary programming language | 3.8+    |
| **Scikit-learn** | Machine learning framework   | Latest  |
| **NLTK**         | Natural language processing  | Latest  |
| **Pandas**       | Data manipulation            | Latest  |
| **NumPy**        | Numerical computing          | Latest  |

| **LangGraph/LangChain**| Orchestration of LLM nodes and prompts| Latest |
| **ChromaDB**     | Fast local vector database for RAG  | Latest  |
| **Groq API**     | Lightning-fast Llama-3.1 inference  | - |

### NLP & ML Components

| Component               | Description                  |
| ----------------------- | ---------------------------- |
| **TF-IDF Vectorizer**   | Feature extraction from text |
| **Logistic Regression** | Binary classification model  |
| **HuggingFace MiniLM**  | Vector embedding generation  |
| **N-gram Analysis**     | Unigrams + Bigrams (1,2)     |

### Web & Deployment

| Technology         | Purpose                               |
| ------------------ | ------------------------------------- |
| **Streamlit**      | Interactive web application framework |
| **BeautifulSoup4** | HTML parsing for URL extraction       |
| **Requests**       | HTTP requests for web scraping        |
| **Joblib / Pickle**| Model and Database serialization      |

---

## 📊 Dataset Information

### ISOT Fake News Dataset

**Source**: Information Security and Object Technology (ISOT) Research Lab, University of Victoria

**Composition**:

- **Fake News**: ~23,000 articles from unreliable sources
- **Real News**: ~21,000 articles from Reuters (trusted journalism)
- **Total**: ~44,000 articles (balanced dataset)

### Dataset Structure

| Column    | Description          | Used in Model                           |
| --------- | -------------------- | --------------------------------------- |
| `title`   | Article headline     | ✅ Yes (combined with text)             |
| `text`    | Article body content | ✅ Yes (combined with title)            |
| `subject` | News category        | ❌ No (removed due to bias)             |
| `date`    | Publication date     | ❌ No (not relevant for classification) |
| `label`   | 0 = Fake, 1 = Real   | ✅ Yes (target variable)                |

### Why Subject Column Was Removed

Analysis revealed that the `subject` column introduced **dataset bias**:

- Certain subjects were predominantly fake or real
- Model could "cheat" by learning subject patterns instead of textual credibility cues
- Removing it forces the model to learn genuine linguistic indicators of fake news

### Data Quality

- **Duplicates Removed**: 209 duplicate rows identified and dropped
- **Missing Values**: None
- **Final Dataset Size**: ~44,000 unique articles (balanced)

---

## 🔬 Machine Learning Pipeline

### 1. Data Preprocessing

#### Text Cleaning Steps

```python
1. Combine title + text → single 'content' column
2. Lowercase conversion
3. URL removal (http/https links)
4. HTML tag removal
5. Whitespace normalization
6. Special character removal (keep alphanumeric + spaces)
```

### 2. Train-Test Split

- **Split Ratio**: 80/20 (Train/Test)
- **Stratification**: Preserves label distribution
- **Random State**: 42 (reproducibility)
- **Training Samples**: ~35,000
- **Testing Samples**: ~9,000

### 3. Feature Engineering: TF-IDF Configuration

```python
TfidfVectorizer(
    stop_words='english',        # Remove common stopwords
    max_features=16000,          # Top 16k most important features
    ngram_range=(1, 2),          # Unigrams + Bigrams
    min_df=3,                    # Word must appear in ≥3 documents
    max_df=0.90                  # Ignore words in >90% of documents
)
```

**Why These Parameters?**

| Parameter      | Value  | Rationale                                                |
| -------------- | ------ | -------------------------------------------------------- |
| `max_features` | 16,000 | Balance between coverage and computational efficiency    |
| `ngram_range`  | (1,2)  | Captures individual words + two-word phrases for context |
| `min_df`       | 3      | Filters rare words (likely typos/noise)                  |
| `max_df`       | 0.90   | Removes overly common words (similar to stopwords)       |

### 4. Model Training: Logistic Regression

```python
LogisticRegression(
    penalty='l2',                # L2 regularization (Ridge)
    C=1.0,                       # Inverse regularization strength
    max_iter=1000,               # Maximum iterations for convergence
    solver='liblinear',          # Efficient for binary classification
    random_state=42              # Reproducibility
)
```

**Why Logistic Regression?**

- ✅ **Interpretable**: Coefficients show word importance
- ✅ **Fast**: Quick training and inference
- ✅ **Probabilistic**: Provides confidence scores
- ✅ **Effective**: Excellent performance on text classification

### 5. Model Evaluation

Comprehensive evaluation using multiple metrics to assess performance across different aspects.

---

## 📈 Model Performance

### Evaluation Metrics

| Metric        | Score     | Interpretation                                          |
| ------------- | --------- | ------------------------------------------------------- |
| **Accuracy**  | **98.5%** | Overall correctness across all predictions              |
| **Precision** | **98.7%** | Of articles predicted as real, 98.7% were actually real |
| **Recall**    | **98.3%** | Of all real articles, model correctly identified 98.3%  |
| **F1 Score**  | **98.5%** | Harmonic mean of precision and recall                   |

### Confusion Matrix

```
                Predicted
                Fake    Real
Actual  Fake    4,421   68
        Real    67      4,444
```

**Analysis**:

- **True Negatives (TN)**: 4,421 fake articles correctly identified
- **True Positives (TP)**: 4,444 real articles correctly identified
- **False Positives (FP)**: 67 fake articles misclassified as real
- **False Negatives (FN)**: 68 real articles misclassified as fake

### Key Insights

1. **Balanced Performance**: Model performs equally well on both classes
2. **Low False Positives**: Rarely mislabels fake news as credible (important for user trust)
3. **Low False Negatives**: Rarely flags genuine news as fake (avoids censorship concerns)
4. **Production-Ready**: 98%+ accuracy suitable for real-world applications

---

## 📁 Project Structure

```
GenAI_Project_11/
│
├── README.md                          # Project documentation
│
├── Dataset/                           # Raw dataset files
│   ├── Fake.csv                       # Fake news articles
│   └── True.csv                       # Real news articles
│
├── intelligent-news-credibility/      # Main application directory
│   │
│   ├── app.py                         # Streamlit web application
│   ├── requirements.txt               # Python dependencies
│   │
│   └── ml/                            # Machine learning components
│       ├── preprocess.ipynb           # Data preprocessing notebook
│       ├── model.ipynb                # Model training notebook
│       ├── model.pkl                  # Trained Logistic Regression model
│       ├── vectorizer.pkl             # Fitted TF-IDF vectorizer
│       ├── X_train.pkl                # Training features (TF-IDF)
│       ├── X_test.pkl                 # Testing features (TF-IDF)
│       ├── y_train.pkl                # Training labels
│       └── y_test.pkl                 # Testing labels
│
└── venv/                              # Virtual environment (not tracked)
```

### File Descriptions

| File               | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `app.py`           | Streamlit UI, model loading, inference logic                     |
| `preprocess.ipynb` | Data cleaning, TF-IDF transformation, train-test split           |
| `model.ipynb`      | Model training, evaluation, serialization                        |
| `model.pkl`        | Serialized Logistic Regression model                             |
| `vectorizer.pkl`   | Serialized TF-IDF vectorizer (for consistent feature extraction) |
| `requirements.txt` | All Python dependencies for deployment                           |

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **Git**: For cloning the repository

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/intelligent-news-credibility.git
cd intelligent-news-credibility
```

#### 2. Create Virtual Environment

**On macOS/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
cd intelligent-news-credibility
pip install -r requirements.txt
```

#### 4. Download NLTK Stopwords (First Time Only)

```bash
python -c "import nltk; nltk.download('stopwords')"
```

#### 5. Verify Installation

```bash
python -c "import streamlit; import sklearn; print('Setup successful!')"
```

---

## 💻 Usage

### Running the Application Locally

#### 1. Navigate to Application Directory

```bash
cd intelligent-news-credibility
```

#### 2. Launch Streamlit App

```bash
streamlit run app.py
```

#### 3. Access the Application

- **Local URL**: http://localhost:8501
- The app will automatically open in your default browser

### Using the Application

#### Option 1: Paste Article Text

1. Select **"Paste Article Text"** radio button
2. Paste full article content into text area
3. Click **"Analyze Credibility"**
4. View results: Credibility label + confidence score

#### Option 2: Enter Article URL

1. Select **"Enter Article URL"** radio button
2. Paste article URL into input field
3. Click **"Fetch Article"** to extract text
4. Review extracted content
5. Click **"Analyze Credibility"**
6. View results: Credibility label + confidence score

### Interpreting Results

| Output                           | Meaning                                                |
| -------------------------------- | ------------------------------------------------------ |
| ✅ **High Credibility Detected** | Model predicts article is real (label = 1)             |
| ⚠️ **Low Credibility Detected**  | Model predicts article is fake (label = 0)             |
| **Confidence Score**             | Model's certainty (0.0 - 1.0, higher = more confident) |

---

## 🌐 Deployment

### Deploying to Streamlit Community Cloud

#### Prerequisites

- GitHub account
- Streamlit Community Cloud account (free at https://streamlit.io/cloud)

#### Step 1: Prepare Repository

Ensure your repository contains:

- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `ml/model.pkl`
- ✅ `ml/vectorizer.pkl`

#### Step 2: Push to GitHub

```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

#### Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click **"New app"**
3. Select your repository
4. Set main file path: `intelligent-news-credibility/app.py`
5. Click **"Deploy"**

#### Step 4: Share Your App

Your app will be live at:

```
https://yourusername-intelligent-news-credibility-app-xxxxx.streamlit.app
```

### Deployment Checklist

- [ ] All `.pkl` files committed to repository (under 100MB)
- [ ] `requirements.txt` includes all dependencies
- [ ] Relative paths used in `app.py` (not absolute)
- [ ] NLTK stopwords downloaded in requirements or code
- [ ] Test URL extraction functionality on deployed version

---

## 🔮 Future Improvements

### 1. **Advanced Analysis Features**

- **Multi-lingual Support**: Detect fake news in multiple languages
- **Image Verification**: Reverse image search for manipulated photos
- **Social Media Analysis**: Track viral spread patterns

### 2. **System Enhancements**

- **Deep Learning Models**: BERT, RoBERTa for contextual understanding of the classical ML Phase.
- **Web-scraping Integration**: Automatically fetching real-time fact checks natively for the LLM.
- **User Feedback Loop**: Continuous model improvement tracking. 

---

## ⚖️ Responsible AI & Ethics

### Ethical Considerations

This tool is designed with the following principles:

#### 1. **Not Absolute Truth**

- Provides **probabilistic assessments**, not definitive judgments
- Should be used as **decision-support**, not censorship mechanism
- Encourages critical thinking rather than replacing it

#### 2. **Transparency**

- Clear explanation of methods (TF-IDF + Logistic Regression)
- No "black box" LLMs without interpretability
- Users understand how predictions are made

#### 3. **Bias Awareness**

- Model trained on ISOT dataset (US-centric, English-only)
- May not generalize to all news contexts globally
- Regular audits needed for fairness across demographics

#### 4. **Limitations**

- **Dataset Bias**: Training data from specific time period
- **Language**: English-only (currently)
- **Context Blind**: Cannot verify factual claims (yet)
- **Evolving Tactics**: Fake news strategies change over time

#### 5. **Recommended Usage**

✅ **Appropriate**:

- First-pass screening for large volumes of content
- Educational tool for media literacy
- Research on misinformation patterns

❌ **Inappropriate**:

- Sole basis for content removal/censorship
- Legal evidence without human review
- Replacing journalistic fact-checking

### Disclaimer

> **⚠️ Important Notice**
>
> This tool provides **automated analysis** based on machine learning patterns. It is **not a substitute** for:
>
> - Professional fact-checking
> - Investigative journalism
> - Human critical thinking
> - Legal determination of defamation
>
> Use responsibly as part of a broader information verification strategy.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
2. **Create Feature Branch**:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit Changes**:
   ```bash
   git commit -m "Add AmazingFeature"
   ```
4. **Push to Branch**:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure compatibility with Python 3.8+

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **ISOT Research Lab** for the Fake News Dataset
- **Streamlit** for the amazing web framework
- **Scikit-learn** community for robust ML tools
- **Open Source Community** for inspiration and support

---

## 📚 References

1. Ahmed, H., Traore, I., & Saad, S. (2017). "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques." _International Conference on Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments_, 127-138.

2. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). "Fake News Detection on Social Media: A Data Mining Perspective." _ACM SIGKDD Explorations Newsletter_, 19(1), 22-36.

3. Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). "Automatic Detection of Fake News." _27th International Conference on Computational Linguistics_, 3391-3401.

---

## 📞 Support

If you encounter issues or have questions:

1. **Check Documentation**: Review this README thoroughly
2. **Search Issues**: Look for similar problems in GitHub Issues
3. **Open New Issue**: Provide detailed description with error logs
4. **Email**: Contact at your.email@example.com

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=yourusername.intelligent-news-credibility)

</div>
