from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from src.config import ModelConfig


def get_model(model_name:str, config: ModelConfig):
    model_registry = {
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "multinomial": MultinomialNB,
        "svn": LinearSVC,
        "xgboost": XGBClassifier,
    }

    return model_registry[model_name](**config.model_dump())