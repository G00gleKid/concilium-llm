# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (required before first run)
pip install -e .

# Run the CLI
concilium -q "Your question here"
concilium -q "..." --tier light|medium|heavy
concilium -q "..." --roles default|archetypes|free|models
concilium -q "..." --show-usage --no-stream

# Pipe input
echo "Your question" | concilium
```

No test suite or linter is configured. There is no build step beyond `pip install -e .`.

## Architecture

The app is a single-pass CLI pipeline — no server, no persistence beyond the `exports/` directory.

**Entry point:** `concilium/main.py:main()` — parses args, loads config, checks model availability, then calls `run()` which orchestrates the full debate pipeline sequentially.

**Pipeline flow:**
1. `ConciliumAgents.generate_experts()` — orchestrator LLM selects 3 expert personas as JSON
2. Debate loop (`max_iterations` rounds) — each expert generates an argument via `generate_argument()`; after each round the orchestrator acts as Moderator via `moderate_round()` and returns a consensus score (stops early if `confidence >= 0.7`)
3. `synthesize()` — orchestrator produces final synthesis + per-expert position JSON
4. Markdown transcript exported to `exports/`

**Key modules:**
- [concilium/agents.py](concilium/agents.py) — all LLM calls; `ConciliumAgents` class owns the OpenAI client and all prompt templates as module-level constants
- [concilium/config.py](concilium/config.py) — `Config` dataclass, `MODEL_TIERS` dict, `load_config()` from TOML, `apply_tier()`
- [concilium/models.py](concilium/models.py) — pure dataclasses: `Expert`, `DebateMessage`, `DebateRound`, `ConsensusResult`
- [concilium/display.py](concilium/display.py) — all Rich console rendering helpers, called from `main.py`

**API layer:** All LLM calls use the OpenAI Python SDK pointed at BotHub (`https://bothub.chat/api/v2/openai/v1`), which provides a unified OpenAI-compatible endpoint for GPT, Claude, and Gemini models. The `BOTHUB_API_KEY` env var is used (falls back to `OPENAI_API_KEY`). There is no model-specific SDK — everything goes through `openai.OpenAI`.

**Configuration precedence:** CLI flags (`--tier`, `--roles`, `--no-stream`, `--show-usage`) override `config.toml`, which overrides `Config` dataclass defaults. The `--tier` flag calls `apply_tier()` and overwrites both orchestrator and all three expert model slots.

**Roles modes** control the expert-generation system prompt only — `default`, `archetypes`, and `free` select different `EXPERT_GEN_SYSTEM*` constants in `agents.py`; `models` skips LLM expert generation entirely and uses the model names as personas.

**JSON parsing robustness:** `_strip_json_fences()` strips markdown code fences, and most LLM calls have a retry or regex fallback if JSON parsing fails — necessary because models occasionally wrap JSON in prose.
