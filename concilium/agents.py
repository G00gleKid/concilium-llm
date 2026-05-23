from __future__ import annotations
import json
import os
import re
from typing import Callable

from openai import OpenAI

from .config import Config
from .models import ConsensusResult, DebateMessage, DebateRound, Expert

LANG_INSTRUCTION = "IMPORTANT: Respond ONLY in the same language as the question. Never switch to another language."


def _lang_instruction(question: str) -> str:
    cyrillic = sum(1 for c in question if 'Ѐ' <= c <= 'ӿ')
    if cyrillic / max(len(question), 1) > 0.1:
        return "IMPORTANT: You MUST respond in Russian. Do not write in English under any circumstances."
    return LANG_INSTRUCTION

EXPERT_GEN_SYSTEM = """\
You are a panel curator for an expert debate system called Concilium.
Given a question, select exactly 3 experts best suited to illuminate \
different dimensions of the question from genuinely distinct epistemic angles.
Each expert must bring a perspective the others do not cover.
Return a JSON object with key "experts" containing an array of 3 objects.
Each object must have: name (string), specialty (string), \
perspective (one-sentence stance), rationale (why selected for this question).
Return ONLY valid JSON, no markdown fences.
""" + LANG_INSTRUCTION

EXPERT_GEN_SYSTEM_ROLES = """\
You are a panel curator for an expert debate system called Concilium.
Given a question, select exactly 3 experts from the following archetypes, \
each bringing a genuinely distinct epistemic angle:
philosopher, economist, engineer, ethicist, scientist, historian, \
psychologist, policy-analyst, entrepreneur, legal-scholar.

Return a JSON object with key "experts" containing an array of 3 objects.
Each object must have: name (string), specialty (string), archetype (one of the list above), \
perspective (one-sentence stance), rationale (why selected for this question).
Return ONLY valid JSON, no markdown fences.
""" + LANG_INSTRUCTION

EXPERT_GEN_SYSTEM_FREE = """\
You are a panel curator for an expert debate system called Concilium.
Given a question, choose exactly 3 people whose perspectives would make the debate \
as illuminating and surprising as possible. There are no constraints on who they can be — \
a fisherman, a recovering addict, a medieval historian, a child psychologist, \
a stand-up comedian, anyone. The only rule: each person must bring something \
genuinely different that the others cannot.
Return a JSON object with key "experts" containing an array of 3 objects.
Each object must have: name (string), specialty (string), \
perspective (one-sentence stance), rationale (why selected for this question).
Return ONLY valid JSON, no markdown fences.
""" + LANG_INSTRUCTION

DEBATE_SYSTEM = LANG_INSTRUCTION + """\

You are an expert panel member in the Concilium debate system.
Your goal: argue rigorously from your unique perspective, challenge other experts by name \
when you disagree, defend your position with evidence and reasoning, and move toward \
nuanced understanding rather than entrenchment.
Write in first person. Keep each argument to 3–5 focused paragraphs.
Do not summarize the conversation — build on it."""

DEBATE_SYSTEM_MODELS = LANG_INSTRUCTION + """\

You are a large language model participating in a panel debate in the Concilium system.
Do not roleplay as a human or adopt any persona — speak as yourself, as an AI language model.
Engage with the question from your own perspective as an AI: your training, your reasoning, \
your uncertainty. Challenge the other models by name when you disagree. \
Be direct, intellectually honest, and curious.
Write in first person. Keep each argument to 3–5 focused paragraphs.
Do not summarize the conversation — build on it."""

MODERATOR_SYSTEM = """\
You are the Moderator of Concilium, an expert debate system.
Your job: assess whether the panel of experts has reached meaningful consensus on the question.

consensus_reached must be false in round 1, always — the first round is for establishing \
positions, not resolving them.

Set consensus_reached to true only when ALL THREE of the following signals are present:
1. Key facts aligned: experts have explicitly agreed on the foundational facts or premises \
underlying the discussion.
2. Position shift: at least one expert has visibly updated their argument in response to \
another's counterargument — not just acknowledged it, but incorporated it (e.g. "I initially \
thought X, but your point about Y changes my view on...").
3. No ambivalence: every expert has clearly stated either why they now agree, or why their \
remaining disagreement is principled and non-blocking — no vague or unresolved tensions left.

If experts merely understand each other's positions without any of the above, \
consensus_reached must remain false.
Respond with a JSON object only. Keys: consensus_reached (boolean), \
confidence (number 0.0–1.0), note (string, max 2 sentences), \
converging_points (array of strings, max 3 items), \
remaining_tensions (array of strings, max 3 items).
Return ONLY valid JSON, no markdown fences.
""" + LANG_INSTRUCTION

SYNTHESIS_SYSTEM = LANG_INSTRUCTION + """\

You are the final synthesizer of Concilium, an expert debate system.
Given the full debate transcript, produce:
1. A comprehensive, nuanced final answer that weighs the strongest arguments from all experts.
   Write in clear, readable prose — avoid heavy jargon and overly academic phrasing. \
   The synthesis should be accessible to an educated general reader, not just specialists.
2. A brief summary of each expert's final position — whether they reached consensus or not.
Return a JSON object with keys:
  "synthesis": string — the weighted final answer (4–8 paragraphs),
  "expert_positions": array of objects, one per expert, with keys:
    "expert_name": string,
    "agreed": boolean — true if the expert broadly agreed with the consensus,
    "summary": string — 2–4 sentences summarising their final stance.
Return ONLY valid JSON, no markdown fences.
""" + LANG_INSTRUCTION


def check_models(config: Config) -> list[tuple[str, str]]:
    """Return list of (model, error) for each unavailable model."""
    from openai import OpenAI, APIError, AuthenticationError, NotFoundError
    client = OpenAI(
        api_key=os.environ.get("BOTHUB_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        base_url=config.api_base_url,
    )
    models_to_check = {config.orchestrator_model} | set(config.expert_models)
    failures: list[tuple[str, str]] = []
    for model in models_to_check:
        try:
            client.chat.completions.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": "hi"}],
            )
        except NotFoundError:
            failures.append((model, "model not found"))
        except AuthenticationError:
            failures.append((model, "authentication failed"))
        except APIError as e:
            failures.append((model, str(e)))
        except Exception as e:
            failures.append((model, str(e)))
    return failures


def _flatten_messages(rounds: list[DebateRound]) -> list[DebateMessage]:
    return [msg for r in rounds for msg in r.messages]


def _build_history_text(history: list[DebateMessage]) -> str:
    parts = []
    for msg in history:
        parts.append(f"[Round {msg.round_number}] {msg.expert.name}: {msg.content}")
    return "\n\n".join(parts)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()
    # If still not valid JSON, try to extract the outermost {...} block
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return text.strip()


class ConciliumAgents:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = OpenAI(
            api_key=os.environ.get("BOTHUB_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
            base_url=config.api_base_url,
        )
        self._usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _accumulate_usage(self, usage) -> None:
        if usage is None:
            return
        self._usage["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self._usage["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    def compute_round_variance(self, round: "DebateRound") -> float:
        """Mean pairwise cosine distance between expert positions in a round."""
        import numpy as np

        texts = [msg.content for msg in round.messages]
        if len(texts) < 2:
            return 0.0

        response = self._client.embeddings.create(
            model=self._config.embedding_model,
            input=texts,
        )
        self._accumulate_usage(response.usage)

        vecs = np.array([e.embedding for e in response.data], dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        n = len(vecs)
        distances = [
            1.0 - float(vecs[i] @ vecs[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return float(np.mean(distances))

    def generate_experts(self, question: str) -> list[Expert]:
        if self._config.roles_mode == "models":
            return [
                Expert(
                    name=model,
                    specialty="language model",
                    perspective="",
                    rationale="",
                    model=model,
                )
                for model in self._config.expert_models
            ]

        if self._config.roles_mode == "archetypes":
            system = EXPERT_GEN_SYSTEM_ROLES
        elif self._config.roles_mode == "free":
            system = EXPERT_GEN_SYSTEM_FREE
        else:
            system = EXPERT_GEN_SYSTEM

        response = self._client.chat.completions.create(
            model=self._config.orchestrator_model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}"},
            ],
        )
        self._accumulate_usage(response.usage)
        choice = response.choices[0]
        raw = _strip_json_fences(choice.message.content or "")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Retry once, asking for compact JSON to avoid hitting token limits
            retry_response = self._client.chat.completions.create(
                model=self._config.orchestrator_model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": (
                        f"Question: {question}\n\n"
                        "Keep all string values concise (under 25 words each). "
                        "Return ONLY a valid JSON object with key \"experts\", no other text."
                    )},
                ],
            )
            self._accumulate_usage(retry_response.usage)
            raw = _strip_json_fences(retry_response.choices[0].message.content or "")
            data = json.loads(raw)
        experts = []
        for i, e in enumerate(data["experts"]):
            model = self._config.expert_models[i] if i < len(self._config.expert_models) else self._config.orchestrator_model
            experts.append(
                Expert(
                    name=e["name"],
                    specialty=e["specialty"],
                    perspective=e["perspective"],
                    rationale=e["rationale"],
                    model=model,
                    archetype=e.get("archetype"),
                )
            )
        return experts

    def generate_argument(
        self,
        expert: Expert,
        question: str,
        history: list[DebateMessage],
        other_experts: list[Expert],
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> DebateMessage:
        round_num = (history[-1].round_number + 1) if history else 1
        others = [e.name for e in other_experts if e.name != expert.name]

        history_text = (
            _build_history_text(history)
            if history
            else "This is the opening round — no prior arguments yet."
        )

        if self._config.roles_mode == "models":
            debate_system = DEBATE_SYSTEM_MODELS
            identity_block = (
                f"You are {expert.name}.\n"
                f"Other models in this debate: {', '.join(others)}"
            )
        else:
            debate_system = DEBATE_SYSTEM
            identity_block = (
                f"You are {expert.name}, {expert.specialty}.\n"
                f"Your epistemic stance: {expert.perspective}\n\n"
                f"Other panel members: {', '.join(others)}"
            )

        messages = [
            {"role": "system", "content": debate_system},
            {
                "role": "user",
                "content": (
                    f"Identity:\n{identity_block}\n\n"
                    f"Question before the panel:\n{question}\n\n"
                    f"Debate transcript so far:\n{history_text}\n\n"
                    f"Now provide your argument for Round {round_num}. "
                    "Engage directly with the arguments above. "
                    "Challenge specific claims by name. Defend your position. "
                    + _lang_instruction(question)
                ),
            },
        ]

        collected: list[str] = []

        if self._config.stream and on_token is not None:
            with self._client.chat.completions.create(
                model=expert.model,
                max_tokens=self._config.max_tokens,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            ) as stream:
                for chunk in stream:
                    if chunk.usage:
                        self._accumulate_usage(chunk.usage)
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        collected.append(delta)
                        on_token(delta)
        else:
            response = self._client.chat.completions.create(
                model=expert.model,
                max_tokens=self._config.max_tokens,
                messages=messages,
            )
            self._accumulate_usage(response.usage)
            collected = [response.choices[0].message.content or ""]

        content = "".join(collected)
        return DebateMessage(
            expert=expert,
            round_number=round_num,
            content=content,
            addressed_to=others,
        )

    def moderate_round(
        self,
        question: str,
        experts: list[Expert],
        rounds: list[DebateRound],
    ) -> tuple[bool, float, str, str]:
        expert_profiles = "\n".join(
            f"- {e.name} ({e.specialty}): {e.perspective}" for e in experts
        )
        transcript = _build_history_text(_flatten_messages(rounds))

        response = self._client.chat.completions.create(
            model=self._config.orchestrator_model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": MODERATOR_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Experts:\n{expert_profiles}\n\n"
                        f"Full debate transcript:\n{transcript}\n\n"
                        "Assess whether consensus has been reached."
                    ),
                },
            ],
        )
        self._accumulate_usage(response.usage)

        raw = _strip_json_fences(response.choices[0].message.content or "{}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    return False, 0.0, "", raw
            else:
                return False, 0.0, "", raw

        return (
            bool(data.get("consensus_reached", False)),
            float(data.get("confidence", 0.0)),
            str(data.get("note", "")),
            raw,
        )

    def synthesize(
        self,
        question: str,
        experts: list[Expert],
        rounds: list[DebateRound],
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> ConsensusResult:
        expert_profiles = "\n".join(
            f"- {e.name} ({e.specialty}): {e.perspective}" for e in experts
        )
        transcript = _build_history_text(_flatten_messages(rounds))

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Experts:\n{expert_profiles}\n\n"
                    f"Full debate transcript:\n{transcript}\n\n"
                    f"Expert names to use exactly in expert_positions: {[e.name for e in experts]}\n\n"
                    f"{_lang_instruction(question)} Write ALL text values in the JSON (synthesis, summaries) in that language — do not switch to English even if the transcript is in English.\n\n"
                    "Produce the final synthesis JSON."
                ),
            },
        ]

        collected: list[str] = []

        if self._config.stream and on_token is not None:
            with self._client.chat.completions.create(
                model=self._config.orchestrator_model,
                max_tokens=self._config.max_tokens,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            ) as stream:
                for chunk in stream:
                    if chunk.usage:
                        self._accumulate_usage(chunk.usage)
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        collected.append(delta)
                        on_token(delta)
        else:
            response = self._client.chat.completions.create(
                model=self._config.orchestrator_model,
                max_tokens=self._config.max_tokens,
                messages=messages,
            )
            self._accumulate_usage(response.usage)
            collected = [response.choices[0].message.content or ""]

        raw = _strip_json_fences("".join(collected))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
            else:
                raise ValueError(f"synthesize() returned invalid JSON: {raw[:200]}")

        expert_by_name = {e.name: e for e in experts}
        expert_positions: list[tuple[Expert, str, bool]] = []
        for pos in data.get("expert_positions", []):
            expert = expert_by_name.get(pos["expert_name"])
            if expert:
                expert_positions.append((expert, pos["summary"], bool(pos.get("agreed", False))))

        # fallback: if model returned old-style dissents key
        if not expert_positions:
            for d in data.get("dissents", []):
                expert = expert_by_name.get(d["expert_name"])
                if expert:
                    expert_positions.append((expert, d["dissent"], False))

        return ConsensusResult(
            reached=True,
            confidence=1.0,
            synthesis=data["synthesis"],
            expert_positions=expert_positions,
            rounds_taken=len(rounds),
            usage=dict(self._usage),
        )
