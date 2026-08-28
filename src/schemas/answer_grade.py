from pydantic import BaseModel, Field


class AnswerGrade(BaseModel):
    groundedness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)