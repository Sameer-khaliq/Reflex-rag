from pydantic import BaseModel, Field


class RewriteOutput(BaseModel):
    rewritten_query: str = Field(min_length=1)