import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


from sklearn.base import BaseEstimator
from typing import SupportsFloat, Union, Dict, Optional, Literal

# Define types for clearer code
ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]

def _get_metrics(
    trained_model: BaseEstimator, 
    X: ArrayLike, 
    y: ArrayLike, 
    split: str = "train", 
    comments: str = "",
    vectorizer_type: Literal["tfidf", "bow"] = "tfidf", 
    vectorizer_params: Optional[dict] = None,
) -> Dict[str, Union[str, float]]:
    """
    Internal function to calculate metrics for a single split.
    
    Args:
        trained_model: A fitted sklearn model.
        X: Vectorized feature matrix (DataFrame, Series or Numpy Array).
        y: Target vector (Series or Numpy Array).
        split: Label for the data split (e.g., 'Train', 'Test').
        comments: Optional user notes about the experiment run.
        vectorizer_type: Type of vectorizer used. Must be 'tfidf' or 'bow'.
            Defaults to 'tfidf'.
        vectorizer_params: Optional dictionary of vectorizer hyperparameters
            (e.g., ngram_range, max_features, max_df). Keys are added
            dynamically as columns in the output row. Defaults to None.

    Returns:
        dict: A dictionary containing all calculated metrics, comments, and any
            additional vectorizer parameters passed.
    """
    # Generate predictions
    y_pred = trained_model.predict(X)
    vp = vectorizer_params or {}

    # Calculate metrics
    row = {
        "model":      trained_model.__class__.__name__,
        "split":      split,
        "vectorizer": vectorizer_type,
        "accuracy":   accuracy_score(y, y_pred),
        "precision":  precision_score(y, y_pred),
        "recall":     recall_score(y, y_pred),
        "f1":         f1_score(y, y_pred),
        "comments":   comments,
    }

    row.update(vp)
    
    return row

def add_new_metrics(
    metrics_df: pd.DataFrame,
    trained_model: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    split: str = "train",
    comments: str = "",
    vectorizer_type: Literal["tfidf", "bow"] = "tfidf", 
    vectorizer_params: Optional[dict] = None,  
) -> pd.DataFrame:
    """
    Calculates metrics and appends them to the tracking DataFrame.
    
    Args:
        metrics_df: The existing DataFrame to update.
        trained_model: A fitted sklearn model.
        X: Vectorized feature matrix (DataFrame, Series or Numpy Array).
            Must be transformed using the same vectorizer used during
            training — never pass raw text.
        y: Target vector (Series or Numpy Array).
        split: "train" or "test".
            Defaults to 'train'.
        comments: Optional user notes about the experiment run.
            Defaults to empty string.
        vectorizer_type: Type of vectorizer used. Must be 'tfidf' or 'bow'.
            Defaults to 'tfidf'.
        vectorizer_params: Optional dictionary of vectorizer hyperparameters
            (e.g., ngram_range, max_features, max_df). Passed through to
            _get_metrics and added as dynamic columns. Defaults to None.

        
    Returns:
        pd.DataFrame: A new DataFrame with the experiment row appended.
            The original metrics_df is not modified in place — always
            reassign the result: metrics_df = add_new_metrics(...).
    """
    
    # Get the metrics dictionary
    new_row_dict = _get_metrics(trained_model,
                                X, y,
                                split,
                                comments,
                                vectorizer_type,
                                vectorizer_params)
    
    new_row = pd.DataFrame([new_row_dict])

    return pd.concat([metrics_df, new_row], ignore_index=True)