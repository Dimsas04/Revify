from pydantic import BaseModel, Field

class FeatureExtractionResult(BaseModel):
    product_type: str
    features: list[str]

class KeyPoint(BaseModel):
    point: str
    frequency: int

class FeatureAnalysis(BaseModel):
    feature: str
    sentiment: str
    key_points: list[KeyPoint] = Field(default_factory=list)
    verdict: str

class FinalAnalysisResult(BaseModel):
    analyses: list[FeatureAnalysis]