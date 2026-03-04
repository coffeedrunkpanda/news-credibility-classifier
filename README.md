# News Credibility Classifier

This repository contains an NLP pipeline designed to distinguish between real and fake news headlines using various machine learning architectures, ranging from traditional linear models to modern Transformer-based fine-tuning.

- [News Credibility Classifier](#news-credibility-classifier)
  - [📖 Project Overview](#-project-overview)
    - [🏆 Key Results \& Findings](#-key-results--findings)
    - [📊 Dataset Description](#-dataset-description)
  - [🛠️ Technologies Used](#️-technologies-used)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Project Structure](#project-structure)
  - [👤 Authors](#-authors)

## 📖 Project Overview

The objective was to build a robust binary classifier capable of identifying "Fake News" based solely on textual headlines.

### 🏆 Key Results & Findings

While initial experiments focused on Scikit-Learn classifiers, the project was recently updated to include **Transformer-based Transfer Learning**. Fine-tuning **DistilBERT** (a distilled version of BERT) significantly outperformed traditional methods by capturing deeper semantic relationships in the headlines.

| Model | Vectorizer / Architecture | Accuracy | F1-Score / Loss |
| :--- | :--- | :--- | :--- |
| **DistilBERT (Fine-tuned)** | **AutoTokenizer (Transformer)** | **98.04%** | **0.0706 (Loss)** |
| LinearSVC | TfidfVectorizer | 94.17% | 0.9416 |
| Logistic Regression | TfidfVectorizer | 94.14% | 0.9413 |
| XGBClassifier | TfidfVectorizer | 93.30% | 0.9328 |
| Multinomial NB | CountVectorizer | 93.10% | 0.9308 |

**Key Takeaways:**
- **Transformers vs. Linear Models:** The jump from 94% to 98% accuracy demonstrates the power of contextual embeddings. Unlike TF-IDF, DistilBERT understands word order and nuance [web:4].
- **Efficiency:** Using `distilbert-base-uncased` allowed for high performance with a smaller memory footprint compared to full BERT, completing 3 epochs with an evaluation speed of ~9182 samples/sec.
- **Modern NLP Pipeline:** The latest iteration utilizes the Hugging Face ecosystem (`Transformers` library), implementing `DataCollatorWithPadding` for efficient dynamic batching during training.

### 📊 Dataset Description

The model was trained on a balanced dataset of news headlines:
- **Total Samples:** 34,152 headlines.
- **Class 0 (Fake News):** 17,572 headlines.
- **Class 1 (Real News):** 16,580 headlines.
- **Preprocessing:** For Transformers, we use the `AutoTokenizer` for `distilbert-base-uncased`. Traditional models used lemmatization and Regex cleaning.

## 🛠️ Technologies Used

- **Deep Learning:** Hugging Face Transformers (`AutoModelForSequenceClassification`, `AutoTokenizer`), PyTorch/TensorFlow.
- **Machine Learning:** Scikit-Learn (Pipelines, LinearSVC, LogisticRegression), XGBoost.
- **NLP:** DistilBERT, NLTK (Lemmatization), TF-IDF.
- **Core:** Python 3.12, Pandas, NumPy.

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher.
- GPU recommended for running `notebooks/modern_nlp.ipynb`.

### Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:coffeedrunkpanda/news-credibility-classifier.git
   cd news-credibility-classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the environment:**
   Download the [dataset and best-trained models](https://drive.google.com/drive/folders/16oOSuR9Eu5ZvbFFjhFPARD46LB_cqf33?usp=share_link) and place them in the following structure:
   - CSVs go into `./data/`
   - `.joblib` files go into `./outputs/models/`

### Project Structure
```text
├── notebooks/          # EDA and model experimentation
├── src/                # Modular Python scripts for the ML pipeline
├── outputs/            # Saved models and evaluation artifacts
├── reports/            # Final project documentation (PDF)
└── scripts/            # Automation for hyperparameter optimization
```

## 👤 Authors

Built with passion by [@coffeedrunkpanda](https://github.com/coffeedrunkpanda) and [@harmandeep2993](https://github.com/harmandeep2993/) during the Ironhack Bootcamp. We combined independent research to compare various NLP methodologies.