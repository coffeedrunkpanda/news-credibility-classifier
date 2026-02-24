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
        X: Feature matrix (DataFrame or Numpy Array).
        y: Target vector (Series or Numpy Array).
        split: Label for the data split (e.g., 'Train', 'Test').
        comments: User notes.

    Returns:
        dict: A dictionary containing all calculated metrics.
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
        X: Feature matrix.
        y: Target vector.
        split: "Train" or "Test".
        comments: Notes about this run.
        
    Returns:
        pd.DataFrame: The updated DataFrame with the new row.
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