from typing import List

from pydantic import BaseModel, Field


class ConfidencePair(BaseModel):
    value: float = Field(description="Value of the requested field.")
    page: int = Field(description="Page number where the value was found")
    

class RiskAnalysis(BaseModel):
    title: str = Field(description="The name of the risk factor")
    discussion: str = Field(description="A 3-4 sentence summary of the risk and its evolution")

class RiskAnalysisList(BaseModel):
    items: List[RiskAnalysis]

class KeywordAnalysis(BaseModel):
    title: str = Field(description="The name of the concern")
    excerpt: str = Field(description="A specific 2-3 sentence excerpt from the text, citing year and page, that indicates the risk")
    description: str = Field(description="A 3-4 sentence summary of the concern and evolution over time")
    impact: str = Field(description="A 1-2 sentence justification of the impact of this concern")

class KeywordAnalysisList(BaseModel):
    items: List[KeywordAnalysis]

