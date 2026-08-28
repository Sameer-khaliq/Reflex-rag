import pytest
from pydantic import ValidationError

from schemas.answer_grade import AnswerGrade
from schemas.chunk_grade import ChunkGrade
from schemas.rewrite import RewriteOutput

MALFORMED_CHUNK_GRADE_JSON = '{"chunk_id": "c1", "grade": "MAYBE"}'
MALFORMED_ANSWER_GRADE_JSON = '{"groundedness_score": "high", "relevance_score": 0.8}'
MALFORMED_ANSWER_GRADE_OUT_OF_RANGE_JSON = '{"groundedness_score": 1.4, "relevance_score": 0.8}'
MALFORMED_REWRITE_JSON = '{"rewritten_query": ""}'
NOT_EVEN_JSON = "the model just returned a sentence instead of JSON"


def test_chunk_grade_rejects_invalid_enum_value():
    with pytest.raises(ValidationError) as exc_info:
        ChunkGrade.model_validate_json(MALFORMED_CHUNK_GRADE_JSON)
    assert "grade" in str(exc_info.value)


def test_answer_grade_rejects_non_numeric_score():
    with pytest.raises(ValidationError) as exc_info:
        AnswerGrade.model_validate_json(MALFORMED_ANSWER_GRADE_JSON)
    assert "groundedness_score" in str(exc_info.value)


def test_answer_grade_rejects_out_of_range_score():
    with pytest.raises(ValidationError) as exc_info:
        AnswerGrade.model_validate_json(MALFORMED_ANSWER_GRADE_OUT_OF_RANGE_JSON)
    assert "groundedness_score" in str(exc_info.value)


def test_rewrite_output_rejects_empty_string():
    with pytest.raises(ValidationError) as exc_info:
        RewriteOutput.model_validate_json(MALFORMED_REWRITE_JSON)
    assert "rewritten_query" in str(exc_info.value)


def test_chunk_grade_rejects_non_json_text():
    with pytest.raises(ValidationError):
        ChunkGrade.model_validate_json(NOT_EVEN_JSON)