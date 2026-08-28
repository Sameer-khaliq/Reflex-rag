from typing import Literal

from pydantic import BaseModel


class ChunkGrade(BaseModel):
    chunk_id: str
    grade: Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]