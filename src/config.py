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

    return AppConfig(
        settings=settings,
        thresholds=thresholds,
        model_tiers=model_tiers,
        providers=providers,
    )