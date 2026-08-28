"""
Central settings module — single import point for every threshold, model
tier, and rate-limit value used across the app. Nothing downstream reads
env vars or YAML directly; everything goes through get_config().

Fails fast at import/boot time if anything required is missing — per
IMPLEMENTATION_PLAN.md §3's error taxonomy: "Config validation failure at
boot -> fail fast. Never silently fall back to a default for a threshold
that should have been explicitly set."
"""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


# ---------------------------------------------------------------------------
# .env-sourced secrets / connection strings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str
    qdrant_collection_name: str

    groq_api_key: str
    openrouter_api_key: str

    tavily_api_key: str

    # Embeddings are the one non-Groq/OpenRouter provider dependency in
    # the stack, isolated strictly to embedding calls — never used for
    # any LLM call site (see model_tiers below, which is Groq/OpenRouter
    # only).
    google_api_key: str


# ---------------------------------------------------------------------------
# thresholds.yaml
# ---------------------------------------------------------------------------
class ThresholdsConfig(BaseModel):
    p_correct_threshold: float
    groundedness_threshold: float
    relevance_threshold: float
    max_iterations: int
    max_generation_attempts: int


# ---------------------------------------------------------------------------
# rate_limits.yaml
# ---------------------------------------------------------------------------
class ModelRef(BaseModel):
    """A single (provider, model slug) pair — enough for llm_clients/ to
    know which base URL + API key to use and what model string to send."""
    provider: str   # "groq" | "openrouter"
    model: str


class ModelSlugPair(BaseModel):
    primary: ModelRef
    fallback: ModelRef


class ModelTierConfig(BaseModel):
    tier1_grading: ModelSlugPair       # document grading, groundedness, relevance, fallback grading (FR-4/7/8/10/11)
    tier2_rewriting: ModelSlugPair     # query rewriting (FR-6)
    tier3_generation: ModelSlugPair    # answer generation (FR-9/FR-18)


class GroqModelLimits(BaseModel):
    rpm: int
    rpd: int
    tpm: int
    tpd: int


class GroqProviderConfig(BaseModel):
    base_url: str
    rate_limits: dict[str, GroqModelLimits]   # keyed by model slug


class OpenRouterProviderConfig(BaseModel):
    base_url: str
    rate_limits: dict  # {"account": {rpm, rpd_unfunded, rpd_funded}}


class ProvidersConfig(BaseModel):
    groq: GroqProviderConfig
    openrouter: OpenRouterProviderConfig


class ResilienceConfig(BaseModel):
    """Shared retry/backoff policy (NFR-10) — every external call site
    (LLM providers, Qdrant, Tavily) reads from this one place rather than
    each hardcoding its own retry count."""
    max_retries: int
    base_delay_s: float


# ---------------------------------------------------------------------------
# retrieval.yaml (new — Phase 2)
# ---------------------------------------------------------------------------
class EmbeddingConfig(BaseModel):
    """Pins model/version/dimension in one place (Risk Register #5) so
    ingestion-time and query-time embedding calls can never silently
    drift apart."""
    provider: str   # "gemini" — only one supported right now, kept
                     # explicit rather than assumed so a future provider
                     # swap is a visible config change, not a silent one
    model: str
    version: str
    dimension: int


class ChunkingConfig(BaseModel):
    min_tokens: int
    max_tokens: int
    overlap_min_pct: float
    overlap_max_pct: float


class RerankConfig(BaseModel):
    top_k: int
    candidate_pool: int
    model: str
    max_candidate_chars: int
    timeout_s: float


class RetrievalConfig(BaseModel):
    sparse_top_n: int
    dense_top_n: int
    rrf_k: int
    bm25_index_dir: str
    chunking: ChunkingConfig
    rerank: RerankConfig


# ---------------------------------------------------------------------------
# YAML loading — raises loudly on anything missing/malformed
# ---------------------------------------------------------------------------
def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required config file missing: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError(f"Config file is empty or invalid: {path}")
    return data


class AppConfig(BaseModel):
    settings: Settings
    thresholds: ThresholdsConfig
    model_tiers: ModelTierConfig
    providers: ProvidersConfig
    resilience: ResilienceConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig


@lru_cache
def get_config() -> AppConfig:
    settings = Settings()

    raw_thresholds = _load_yaml("thresholds.yaml")
    thresholds = ThresholdsConfig(
        p_correct_threshold=raw_thresholds["document_grading"]["p_correct_threshold"],
        groundedness_threshold=raw_thresholds["answer_grading"]["groundedness_threshold"],
        relevance_threshold=raw_thresholds["answer_grading"]["relevance_threshold"],
        max_iterations=raw_thresholds["iteration_caps"]["max_iterations"],
        max_generation_attempts=raw_thresholds["iteration_caps"]["max_generation_attempts"],
    )

    raw_limits = _load_yaml("rate_limits.yaml")

    def _slug_pair(entry: dict) -> ModelSlugPair:
        return ModelSlugPair(
            primary=ModelRef(**entry["primary"]),
            fallback=ModelRef(**entry["fallback"]),
        )

    model_tiers = ModelTierConfig(
        tier1_grading=_slug_pair(raw_limits["models"]["tier1_grading"]),
        tier2_rewriting=_slug_pair(raw_limits["models"]["tier2_rewriting"]),
        tier3_generation=_slug_pair(raw_limits["models"]["tier3_generation"]),
    )

    raw_providers = raw_limits["providers"]
    providers = ProvidersConfig(
        groq=GroqProviderConfig(
            base_url=raw_providers["groq"]["base_url"],
            rate_limits={
                slug: GroqModelLimits(**limits)
                for slug, limits in raw_providers["groq"]["rate_limits"].items()
            },
        ),
        openrouter=OpenRouterProviderConfig(
            base_url=raw_providers["openrouter"]["base_url"],
            rate_limits=raw_providers["openrouter"]["rate_limits"],
        ),
    )

    resilience = ResilienceConfig(**raw_limits["resilience"])

    raw_retrieval = _load_yaml("retrieval.yaml")
    embedding = EmbeddingConfig(**raw_retrieval["embedding"])
    retrieval = RetrievalConfig(
        sparse_top_n=raw_retrieval["retrieval"]["sparse_top_n"],
        dense_top_n=raw_retrieval["retrieval"]["dense_top_n"],
        rrf_k=raw_retrieval["retrieval"]["rrf_k"],
        bm25_index_dir=raw_retrieval["retrieval"]["bm25_index_dir"],
        chunking=ChunkingConfig(**raw_retrieval["chunking"]),
        rerank=RerankConfig(**raw_retrieval["rerank"]),
    )

    return AppConfig(
        settings=settings,
        thresholds=thresholds,
        model_tiers=model_tiers,
        providers=providers,
        resilience=resilience,
        embedding=embedding,
        retrieval=retrieval,
    )