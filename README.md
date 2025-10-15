# 📚 Semantic Book Recommender System

An intelligent **book recommendation and sentiment analysis system** built with **LangChain**, **Gradio**, and **Hugging Face Spaces**.  
This project combines **semantic text embeddings**, **vector search**, and **emotion-based NLP** to recommend books aligned with user preferences and emotional tone.

---

## 🚀 Features

- 🧠 **Semantic Book Search** using OpenAI embeddings + Chroma vector store  
- 💬 **Sentiment Analysis** of book descriptions or user reviews (anger, joy, fear, surprise, etc.)  
- 📖 **Category-based Filtering** (fiction, business, science, etc.)  
- ⚡ **Fast Inference UI** with Gradio and LangChain  
- ☁️ **Deployable on Hugging Face Spaces** via CI/CD  

---

## 🧩 Project structure

| File/Folder | Purpose | Status |
|-------------|---------|--------|
| **`app.py`** | Main Gradio Interface | **Core** |
| **`sentiment_analysis.py`** | Emotion Analysis (6 emotions) | **Core** |
| **`vector_search.py`** | ChromaDB Semantic Search | **Core** |
| **`text_classification/`** | Transformers Pipeline | **Core** |
| **`data/`** | 7k Books | **4.14MB** |
| **`requirements.txt`** | Dependencies | **Pinned** |
| **`.env.example`** | API Keys Template | **Copy me** |
| **`README.md`** | Setup Guide | **You're here** |

## 🛠️ Tech Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **UI Framework** | Gradio | `^4.44.0` | Interactive dashboard with Glass theme |
| **Embeddings** | OpenAI | `text-embedding-3-small` | Semantic text understanding (1536 dims) |
| **Vector Database** | Chroma | `^0.5.0` | Fast similarity search & storage |
| **NLP Pipeline** | Transformers + LangChain | `^4.40.0` | Emotion detection & text processing |
| **Visualization** | Seaborn + Matplotlib | `^0.13.2` | Emotion charts & recommendation graphs |
| **CI/CD** | GitHub Actions + HF Spaces | Latest | Auto-deployment to cloud |
| **Containerization** | Docker | `^24.0` | Local dev & production consistency |
| **Environment** | Python | `3.10+` | Core runtime |
| **Data Processing** | Pandas | `^2.2.0` | CSV handling & preprocessing |
| **API Client** | OpenAI + HF Hub | Latest | Model inference |

**💡 Pro Tip:** All dependencies are pinned for reproducibility!

---

## ⚙️ Quick Start

### Prerequisites
- Python 3.10+
- Git
- [OpenAI API Key](https://platform.openai.com/account/api-keys)
- [Hugging Face Token](https://huggingface.co/settings/tokens)

### 1️⃣ Clone Repository
```bash
git clone https://github.com/thphuc273/semantic-book-recommender.git
cd semantic-book-recommender
```

### 2️⃣ Set up Environment
```bash
# Copy template
cp .env.example .env
```

Edit `.env` file:
```
OPENAI_API_KEY=sk-your-openai-key-here
HF_TOKEN=hf_your-huggingface-token
```

### 3️⃣ Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install everything
pip install -r requirements.txt
```
### 4️⃣ Download Dataset

[Kaggle dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

### 5️⃣ Launch App
```
python app.py
```

✅ Opens: http://127.0.0.1:7860