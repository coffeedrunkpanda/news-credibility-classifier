# News Credibility Classifier

This repository contains an NLP pipeline designed to distinguish between real and fake news headlines using various machine learning architectures.

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

After extensive hyperparameter tuning using `RandomizedSearchCV`, the **Linear Support Vector Classifier (LinearSVC)** paired with **TF-IDF Vectorization** emerged as the top-performing model. While tree-based models like Random Forest achieved perfect training scores, they exhibited signs of overfitting compared to the more generalized linear models.

| Model | Vectorizer | Test Accuracy | Test F1-Score |
| :--- | :--- | :--- | :--- |
| **LinearSVC** | TfidfVectorizer | **94.17%** | **0.9416** |
| Logistic Regression | TfidfVectorizer | 94.14% | 0.9413 |
| XGBClassifier | TfidfVectorizer | 93.30% | 0.9328 |
| Multinomial NB | CountVectorizer | 93.10% | 0.9308 |
| Random Forest | TfidfVectorizer | 92.34% | 0.9233 |

**Key Takeaways:**
- **Linearity in Text:** High-dimensional text data often responds best to linear boundaries (SVC/LogReg), which outperformed complex ensembles like XGBoost in this specific task.
- **Feature Engineering:** TF-IDF consistently yielded better generalization than simple word counts for most models.
- **Overfitting:** The Random Forest model achieved a 1.0 training F1-score but dropped to ~0.92 on test data, suggesting it memorized noise rather than learning underlying patterns.

### 📊 Dataset Description

The model was trained on a balanced dataset of news headlines:
- **Total Samples:** 34,152 headlines.
- **Class 0 (Fake News):** 17,572 headlines.
- **Class 1 (Real News):** 16,580 headlines.
- **Preprocessing:** Applied lemmatization, and character cleaning to reduce noise.

## 🛠️ Technologies Used

- **Core:** Python 3.12, Pandas, NumPy.
- **Machine Learning:** Scikit-Learn (Pipelines, RandomizedSearchCV), XGBoost.
- **NLP:** NLTK (Lemmatization), TF-IDF, Bag of Words.
- **Visualization:** Matplotlib, Seaborn.

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher.

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