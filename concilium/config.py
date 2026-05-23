from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

BOTHUB_BASE_URL = "https://bothub.chat/api/v2/openai/v1"

MODEL_TIERS: dict[str, dict] = {
    "light": {
        "orchestrator": "gemini-2.5-flash-lite",
        "experts": ["gpt-4.1-nano", "claude-haiku-4.5", "gemini-2.5-flash-lite"],
    },
    "medium": {
        "orchestrator": "gemini-2.5-flash",
        "experts": ["gpt-4.1-mini", "claude-sonnet-4.5", "gemini-2.5-flash"],
    },
    "heavy": {
        "orchestrator": "gemini-2.5-pro",
        "experts": ["gpt-5", "claude-sonnet-4.6", "gemini-2.5-pro"],
    },
}

DEFAULT_TIER = "light"
DEFAULT_VARIANCE_DELTA = 0.05


ROLES_MODES = ("default", "archetypes", "free", "models")


@dataclass
class Config:
    max_iterations: int = 4
    roles_mode: str = "default"  # "default" | "archetypes" | "free" | "models"
    orchestrator_model: str = MODEL_TIERS[DEFAULT_TIER]["orchestrator"]
    expert_models: list[str] = field(default_factory=lambda: list(MODEL_TIERS[DEFAULT_TIER]["experts"]))
    max_tokens: int = 4096
    stream: bool = True
    show_usage: bool = False
    api_base_url: str = BOTHUB_BASE_URL
    divergence_detection: bool = True
    divergence_stable_rounds: int = 3
    divergence_variance_delta: float = DEFAULT_VARIANCE_DELTA
    embedding_model: str = "text-embedding-3-small"
    debug: bool = False


def apply_tier(config: Config, tier: str) -> None:
    tier_cfg = MODEL_TIERS[tier]
    config.orchestrator_model = tier_cfg["orchestrator"]
    config.expert_models = list(tier_cfg["experts"])


def load_config(path: str = "config.toml") -> Config:
    config_path = Path(path)
    if not config_path.exists():
        return Config()

    if sys.version_info >= (3, 11):
        import tomllib
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    else:
        try:
            import tomli
            with open(config_path, "rb") as f:
                data = tomli.load(f)
        except ImportError:
            return Config()

    debate = data.get("debate", {})
    model = data.get("model", {})
    display = data.get("display", {})
    experts_cfg = data.get("experts", {})
    divergence = data.get("divergence", {})

    default = MODEL_TIERS[DEFAULT_TIER]
    expert_models = [
        experts_cfg.get("expert_1", default["experts"][0]),
        experts_cfg.get("expert_2", default["experts"][1]),
        experts_cfg.get("expert_3", default["experts"][2]),
    ]

    raw_roles = debate.get("roles_mode", "default")
    # backward compat: enable_roles = true → archetypes
    if raw_roles == "default" and debate.get("enable_roles", False):
        raw_roles = "archetypes"
    roles_mode = raw_roles if raw_roles in ROLES_MODES else "default"

    return Config(
        max_iterations=debate.get("max_iterations", 10),
        roles_mode=roles_mode,
        orchestrator_model=model.get("orchestrator", default["orchestrator"]),
        expert_models=expert_models,
        max_tokens=model.get("max_tokens", 4096),
        stream=display.get("stream", True),
        show_usage=display.get("show_usage", False),
        api_base_url=model.get("api_base_url", BOTHUB_BASE_URL),
        divergence_detection=divergence.get("detection", True),
        divergence_stable_rounds=divergence.get("stable_rounds", 3),
        divergence_variance_delta=divergence.get("variance_delta", DEFAULT_VARIANCE_DELTA),
        embedding_model=divergence.get("embedding_model", "text-embedding-3-small"),
        debug=display.get("debug", False),
    )
