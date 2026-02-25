import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_cleaning import clean_text
from src.preprocessing import lemmatize

def load_data(filename):

    df = pd.read_csv(filename, names = ["raw"])
    df = df.raw.str.split("\t", n = 1, expand=True)
    df.columns = ["labels", "text"]

    df["labels"] = df["labels"].astype("int")
    
    return df

def build_datasets(X, y, test_size = 0.2, random_state = 13):

    X, y = _process_pipeline(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size= test_size,
                                                        random_state=random_state)

    return X_train, X_test, y_train, y_test

def _process_pipeline(text, labels):

    # Clean
    text.apply(clean_text)

    # Preprocess
    X_lemma = text.apply(lemmatize)

    return text, labels
    