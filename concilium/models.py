from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Expert:
    name: str
    specialty: str
    perspective: str
    rationale: str
    model: str = ""              # LLM model assigned to this expert slot
    archetype: str | None = None  # set when roles_mode="archetypes"


@dataclass
class DebateMessage:
    expert: Expert
    round_number: int
    content: str
    addressed_to: list[str] = field(default_factory=list)


@dataclass
class DebateRound:
    round_number: int
    messages: list[DebateMessage] = field(default_factory=list)
    moderator_note: str | None = None
    position_variance: float | None = None  # mean pairwise cosine distance between experts


@dataclass
class ConsensusResult:
    reached: bool
    confidence: float
    synthesis: str
    expert_positions: list[tuple[Expert, str, bool]] = field(default_factory=list)
    rounds_taken: int = 0
    usage: dict[str, int] = field(default_factory=dict)
