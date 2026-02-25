import sys
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent

# Setup
sys.path.append(str(PROJECT_ROOT))  # points to project root

import pandas as pd
from src.data import load_data, build_datasets
from src.preprocessing import setup_nltk_data
from src.feature_extraction import get_vectorizer
from src.evaluate import add_new_metrics

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score

# ============DATASET==============

print("Setup nltk...")
setup_nltk_data()

print("Read Data...")
# Read Data
filename = "data/training_data.csv"

df = load_data(filename)
X = df.text
y = df.labels

print("Create the Datasets...")
# Build Dataset with clean and preprocessing
X_train, X_test, y_train, y_test = build_datasets(X, y)

seed = 13
results_list = []

# ============Hyperparameter Search==============

SEARCH_SPACES = {
    "random_forest": {
        "classifier": RandomForestClassifier(),
        "param_distributions": {
            "vectorizer": [TfidfVectorizer(), CountVectorizer()],
            "vectorizer__ngram_range": [(1,1), (1,2)],
            "vectorizer__max_features": [1000, 5000],
            "classifier__n_estimators": [100, 200, 400],
            
            "classifier__criterion": ["gini", "entropy"],
        },
    },
    "logreg": {
        "classifier": LogisticRegression(max_iter=2000),
        "param_distributions": {
            "vectorizer": [TfidfVectorizer(), CountVectorizer()],
            "vectorizer__ngram_range": [(1,1), (1,2)],
            "vectorizer__max_features": [1000, 5000],

            "classifier__C": [0.1, 1.0, 10.0],
        },
    },
    "xgb": {
        "classifier": XGBClassifier(eval_metric="mlogloss"), # cross-entropy loss
        "param_distributions": {
            "vectorizer": [TfidfVectorizer(), CountVectorizer()],
            "vectorizer__ngram_range": [(1,1), (1,2)],
            "vectorizer__max_features": [1000, 5000],

            "classifier__n_estimators": [100, 200, 400],
            "classifier__max_depth": [3, 6, 9],
            "classifier__learning_rate": [0.05, 0.1, 0.3],
        },
    },
    "svn": {
    "classifier": LinearSVC(),
    "param_distributions": {
        "vectorizer": [TfidfVectorizer(), CountVectorizer()],
        "vectorizer__ngram_range": [(1,1), (1,2)],
        "vectorizer__max_features": [1000, 5000],

        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [0.05, 0.1, 0.3],
    },
    },
    "multinomial": {
        "classifier": MultinomialNB(),
        "param_distributions": {
            "vectorizer": [TfidfVectorizer(), CountVectorizer()],
            "vectorizer__ngram_range": [(1,1), (1,2)],
            "vectorizer__max_features": [1000, 5000],

            # Search space for Multinomial Naive Bayes
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0, 5.0],
            "classifier__fit_prior": [True, False],
        },
    },
}

print("Starting experiments...")

for model_key, spec in SEARCH_SPACES.items():

    pipe = Pipeline([
        ("vectorizer", TfidfVectorizer()), # placeholder
        ("classifier", spec["classifier"]), ## placeholder
    ])

    # Randomized search 
    search = RandomizedSearchCV(
        pipe,
        param_distributions=spec["param_distributions"],
        n_iter=30,                  
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=seed,
    refit=True
    )

    search.fit(X_train, y_train)

   # 2. Extract the best model
    best_pipeline = search.best_estimator_

    # 3. Make predictions on Train and Test
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    # 4. Calculate Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    # 5. Clean parameters for the CSV 
    # (Removes the actual Scikit-Learn objects so the string looks clean in the CSV)
    clean_params = {k: v for k, v in search.best_params_.items() 
                    if k not in ['classifier', 'vectorizer']}

    model_name = best_pipeline.named_steps["classifier"].__class__.__name__

    # 6. Add the Train metrics row
    results_list.append({
        "model": model_name,
        "split": "train",
        "accuracy": train_acc,
        "f1_score": train_f1,
        "best_parameters": str(clean_params)
    })
    
    # 7. Add the Test metrics row
    results_list.append({
        "model": model_name,
        "split": "test",
        "accuracy": test_acc,
        "f1_score": test_f1,
        "best_parameters": str(clean_params)
    })


print("\nExperiments complete. Saving metrics...")

# 8. Convert the list of dicts to a Pandas DataFrame and save it
metrics_df = pd.DataFrame(results_list)

# We use index=False so Pandas doesn't write an extra column of row numbers
metrics_df.to_csv("outputs/best_models_evaluation_metrics.csv", index=False)

# Display the final table in the console
print(metrics_df)