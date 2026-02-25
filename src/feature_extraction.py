from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.config import VectorizerConfig

def get_representation(X, max_features, ngram_range = (1,2) ,repr = "tfidf", **kwargs):

    if repr == "tfidf":
        vectorizer = TfidfVectorizer(max_features = max_features, ngram_range=ngram_range, **kwargs)
        representations = vectorizer.fit_transform(X)
        feature_names = vectorizer.get_feature_names_out()

    # Bag of Words
    else:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range, **kwargs)
        representations = vectorizer.fit_transform(X)
        feature_names = vectorizer.get_feature_names_out()

    return representations, vectorizer, feature_names

# Refactored
def get_vectorizer(vectorizer_type:str, config:VectorizerConfig):

    vectorizer_registry = {
        "tfidf": TfidfVectorizer,
        "bow": CountVectorizer
    }

    return vectorizer_registry[vectorizer_type](**config.model_dump())