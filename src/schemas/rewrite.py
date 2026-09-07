from pydantic import BaseModel, Field


class RewriteOutput(BaseModel):
    rewritten_query: str = Field(
        min_length=3,
        max_length=512,
        description= "Search-optimized rewritten query",)