# Concilium LLM

A CLI tool that assembles a panel of 3 AI experts tailored to your question, runs a dynamic debate between them, and produces a weighted consensus answer — with each expert's final position summarised at the end.

## Install

```bash
pip install -e .
```

Requires Python 3.10+. Add your BotHub API key to a `.env` file in the project root:

```
BOTHUB_API_KEY=your_key_here
```

Get a key at [bothub.chat/profile/for-developers](https://bothub.chat/profile/for-developers). `OPENAI_API_KEY` is used as a fallback if `BOTHUB_API_KEY` is not set.

## Usage

```bash
# Question as argument
concilium -q "Should we colonize Mars before fixing Earth's problems?"

# Pipe from stdin
echo "Is consciousness substrate-independent?" | concilium

# Interactive prompt (no -q flag)
concilium

# Choose model tier (light by default)
concilium -q "..." --tier light
concilium -q "..." --tier medium
concilium -q "..." --tier heavy

# Choose expert selection mode (default by default)
concilium -q "..." --roles default     # guided selection, no archetype constraints
concilium -q "..." --roles archetypes  # pick from fixed list: philosopher, economist, etc.
concilium -q "..." --roles free        # no constraints — anyone, any background
concilium -q "..." --roles models      # no personas — models debate as themselves

# Show token usage at the end
concilium -q "..." --show-usage

# Disable streaming (batch mode)
concilium -q "..." --no-stream

# Custom config file
concilium -q "..." --config /path/to/my_config.toml
```

## Model tiers

Tiers control which models are used for all three expert slots and the orchestrator (moderation + synthesis). Switch with `--tier`.

| Tier | Orchestrator | Expert 1 | Expert 2 | Expert 3 |
|------|-------------|----------|----------|----------|
| `light` (default) | gemini-2.5-flash-lite | gpt-4.1-nano | claude-haiku-4.5 | gemini-2.5-flash-lite |
| `medium` | gemini-2.5-flash | gpt-4.1-mini | claude-sonnet-4.5 | gemini-2.5-flash |
| `heavy` | gemini-2.5-pro | gpt-5 | claude-sonnet-4.6 | gemini-2.5-pro |

Individual models can be overridden in `config.toml` (see below). `--tier` takes priority over `config.toml`.

## Configuration (`config.toml`)

```toml
[debate]
max_iterations = 2        # max debate rounds before forced synthesis
roles_mode = "default"    # "default" | "archetypes" | "free" | "models"

[model]
orchestrator = "gemini-2.5-flash-lite"
max_tokens   = 4096
# api_base_url = "https://bothub.chat/api/v2/openai/v1"  # default (BotHub)

[experts]
expert_1 = "gpt-4.1-nano"
expert_2 = "claude-haiku-4.5"
expert_3 = "gemini-2.5-flash-lite"

[display]
stream     = true    # stream tokens to terminal in real time
show_usage = false   # show token usage stats at end
```

### Expert selection modes (`roles_mode`)

| Mode | Description |
|------|-------------|
| `default` | Orchestrator freely selects 3 experts suited to the question |
| `archetypes` | Experts are chosen from a fixed list: `philosopher`, `economist`, `engineer`, `ethicist`, `scientist`, `historian`, `psychologist`, `policy-analyst`, `entrepreneur`, `legal-scholar` |
| `free` | No constraints — the orchestrator can pick anyone: a fisherman, a stand-up comedian, a recovering addict |
| `models` | No personas at all — the three LLMs debate as themselves, referring to each other by model name |

## How it works

1. **Expert selection** — the orchestrator analyzes your question and generates 3 experts with distinct epistemic angles (name, specialty, perspective, rationale).
2. **Debate loop** — experts take turns arguing and challenging each other by name. After each round, a Moderator assesses consensus (confidence ≥ 0.7 stops the debate early).
3. **Synthesis** — a final synthesis agent weighs all arguments and produces a balanced conclusion in plain language, followed by each expert's final position (agreed / dissent).
4. **Export** — the full transcript and synthesis are saved as Markdown in the `exports/` folder.

## Future extensions (backlog)

- **Fixed rounds mode** — each expert speaks exactly N times, regardless of consensus
- **Fixed expert pool** — predefined roster of experts; system picks the 3 best matches
- **Human arbitration** — a human-in-the-loop moderator who can interject via CLI
- **Web UI** — FastAPI + SSE streaming frontend
