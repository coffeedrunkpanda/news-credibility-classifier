from pydantic import BaseModel, Field
from typing import Literal, Any

class VectorizerConfig(BaseModel):
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 1000
    max_df: float = 1.0

class ModelConfig(BaseModel):
    n_estimators: int = 200
    criterion: str = "entropy"

class ExperimentConfig(BaseModel):
    model_name: str
    model_params: ModelConfig = Field(default_factory=ModelConfig)
    vectorizer_type:Literal["bow", "tfidf"] = "tfidf"
    vectorizer_params: VectorizerConfig = Field(default_factory=VectorizerConfig)
    comments:str

