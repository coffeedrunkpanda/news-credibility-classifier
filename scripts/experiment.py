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

from src.config import ExperimentConfig, VectorizerConfig, ModelConfig
from src.models import get_model

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


# Experiments
# models_vect = ["xgboost"]
# ngram_vect = [(1,1)]
# max_features_vect = [1000]
# max_df_vect = [0.5]
# vectorizer_vect = ["tfidf"]

models_vect = ["random_forest", "logistic_regression", "multinomial", "svn", "xgboost"]
ngram_vect = [(1,1), (1,2), (2,2)]
max_features_vect = [1000, 5000]
max_df_vect = [0.5, 0.8, 1.0]
vectorizer_vect = ["tfidf", "bow"]

metrics_df = pd.DataFrame()

for model_name in models_vect:
    for n_gram_range in ngram_vect:
        for max_features in max_features_vect:
            for max_df in max_df_vect:
                for vectorizer_name in vectorizer_vect:

                    print("Get training configs...")
                    # Feature extraction
                    vectorizer_params = VectorizerConfig(ngram_range= n_gram_range,
                                                        max_features=max_features,
                                                        max_df=max_df)

                    model_params = ModelConfig()

                    config = ExperimentConfig(model_name=model_name,
                                            vectorizer_type=vectorizer_name,
                                            vectorizer_params=vectorizer_params,
                                            model_params=model_params)

                    vectorizer = get_vectorizer(config.vectorizer_type, config.vectorizer_params)
                    vectorizer.fit(X_train)

                    X_train_vector = vectorizer.transform(X_train)

                    print("Train the models...")
                    # Train Model
                    model= get_model(config.model_name, config.model_params)
                    model.fit(X_train_vector,y_train)

                    print("Get Metrics...")
                    # Get Metrics
                    # train metrics

                    metrics_df = add_new_metrics(metrics_df, model, X_train_vector, y_train, "train", "testing metrics", config.vectorizer_type, config.vectorizer_params.model_dump())

                    # test metrics
                    X_test_vector = vectorizer.transform(X_test)
                    metrics_df = add_new_metrics(metrics_df, model, X_test_vector, y_test, "test", "testing test metrics", config.vectorizer_type, config.vectorizer_params.model_dump())


# Save Metrics
metrics_df.to_csv("outputs/metrics.csv")