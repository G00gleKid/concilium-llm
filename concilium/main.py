from __future__ import annotations
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console

from .agents import ConciliumAgents, check_models
from .config import Config, load_config, MODEL_TIERS, ROLES_MODES, apply_tier
from .display import (
    render_dissents,
    render_expert_message,
    render_expert_turn_end,
    render_expert_turn_start,
    render_experts,
    render_header,
    render_moderator_note,
    render_question,
    render_round_header,
    render_synthesis,
    render_usage,
)
from .models import ConsensusResult, DebateRound, Expert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="concilium",
        description="Multi-expert debate system powered by Claude.",
    )
    parser.add_argument(
        "-q", "--question",
        help="The question to debate. Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        metavar="PATH",
        help="Path to config.toml (default: config.toml in current directory).",
    )
    parser.add_argument(
        "--tier",
        choices=list(MODEL_TIERS.keys()),
        default=None,
        help="Model tier: light (default), medium, or heavy.",
    )
    parser.add_argument(
        "--roles",
        choices=list(ROLES_MODES),
        default=None,
        help="Expert selection mode: default (guided), archetypes (fixed list), free (no constraints).",
    )
    parser.add_argument(
        "--no-stream",
        dest="no_stream",
        action="store_true",
        help="Disable streaming output (batch mode).",
    )
    parser.add_argument(
        "--show-usage",
        dest="show_usage",
        action="store_true",
        help="Show token and cache usage at the end.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write a debug log (*_debug.md) alongside the main export.",
    )
    return parser.parse_args()


def _make_slug(question: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", question.lower())
    return re.sub(r"[\s_-]+", "-", slug).strip("-")[:50]


def resolve_question(args: argparse.Namespace, console: Console) -> str:
    if args.question:
        return args.question.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    console.print("[dim]Enter your question (press Enter twice or Ctrl+D when done):[/dim]")
    lines = []
    try:
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        pass
    return "\n".join(lines).strip()


def run(question: str, config: Config, console: Console) -> None:
    agents = ConciliumAgents(config)

    render_header(console)
    render_question(question, console)

    console.print("[dim]Selecting experts for your question…[/dim]")
    experts = agents.generate_experts(question)
    render_experts(experts, console)

    rounds: list[DebateRound] = []
    history = []
    consensus_reached = False
    force_extra_round = False
    debug_lines: list[str] = []

    if config.debug:
        debug_lines.append(f"# Debug Log\n\n**Question:** {question}\n")

    for round_num in range(1, config.max_iterations + 1):
        render_round_header(round_num, console)
        current_round = DebateRound(round_number=round_num)

        for expert in experts:
            if config.stream:
                render_expert_turn_start(expert, experts, console)
                buffer: list[str] = []

                def on_token(text: str, _buf: list[str] = buffer) -> None:
                    _buf.append(text)
                    console.print(text, end="", highlight=False)

                msg = agents.generate_argument(
                    expert, question, history, experts, on_token=on_token
                )
                render_expert_turn_end(console)
            else:
                msg = agents.generate_argument(expert, question, history, experts)
                render_expert_message(expert, experts, msg.content, console)

            current_round.messages.append(msg)
            history.append(msg)

        # Moderation after each round
        console.print("[dim]Moderator is assessing the debate…[/dim]")
        reached, confidence, note, raw_json = agents.moderate_round(question, experts, rounds + [current_round])
        current_round.moderator_note = note

        # Divergence detection: compute embedding variance before appending
        if config.divergence_detection:
            current_round.position_variance = agents.compute_round_variance(current_round)

        rounds.append(current_round)

        render_moderator_note(note, confidence, current_round.position_variance, console)

        if config.debug:
            variance_str = f"{current_round.position_variance:.4f}" if current_round.position_variance is not None else "N/A"
            debug_lines.append(f"## Round {round_num}\n")
            debug_lines.append(f"- **consensus_reached:** {reached}")
            debug_lines.append(f"- **confidence:** {confidence}")
            debug_lines.append(f"- **position_variance:** {variance_str}")
            msg_lengths = ", ".join(f"{msg.expert.name}: {len(msg.content)} chars" for msg in current_round.messages)
            debug_lines.append(f"- **message lengths:** {msg_lengths}")
            debug_lines.append(f"\n**Raw moderator JSON:**\n```json\n{raw_json}\n```\n")

        if reached and confidence >= 0.7:
            consensus_reached = True
            break
        elif reached and confidence < 0.7:
            # Almost there — one more round; skip plateau check this iteration
            console.print("[dim]Near-consensus — continuing one more round for clarity.[/dim]")
            force_extra_round = True
            continue

        # Plateau check: if variance is stable for divergence_stable_rounds consecutive rounds, stop
        if force_extra_round:
            force_extra_round = False

        if config.divergence_detection and len(rounds) >= config.divergence_stable_rounds:
            recent_variances = [
                r.position_variance for r in rounds[-config.divergence_stable_rounds:]
                if r.position_variance is not None
            ]
            if len(recent_variances) == config.divergence_stable_rounds:
                if max(recent_variances) - min(recent_variances) <= config.divergence_variance_delta:
                    console.print("[yellow]Stable divergence detected — positions are not moving. Stopping.[/yellow]")
                    if config.debug:
                        debug_lines.append(f"**Plateau detected at round {round_num}** — variance delta {max(recent_variances) - min(recent_variances):.4f} <= {config.divergence_variance_delta}\n")
                    break

    if not consensus_reached:
        console.print(
            f"[yellow]Max iterations ({config.max_iterations}) reached. Synthesizing from current state.[/yellow]"
        )

    console.print("[dim]Generating final synthesis…[/dim]")

    from rich.status import Status
    with Status("[dim]Synthesizing…[/dim]", console=console):
        result = agents.synthesize(question, experts, rounds)

    result.reached = consensus_reached
    render_synthesis(result, console)
    render_dissents(result, console)

    if config.show_usage:
        render_usage(result, console)

    _export_markdown(question, experts, rounds, result, console)

    if config.debug and debug_lines:
        _export_debug(question, debug_lines, console)


def _export_markdown(
    question: str,
    experts: list[Expert],
    rounds: list[DebateRound],
    result: ConsensusResult,
    console: Console,
) -> None:
    slug = _make_slug(question)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    path = exports_dir / f"{timestamp}_{slug}.md"

    lines: list[str] = []
    lines.append(f"# Concilium Debate\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    lines.append(f"**Question:** {question}\n\n")

    lines.append("## Expert Panel\n")
    for expert in experts:
        lines.append(f"### {expert.name}")
        lines.append(f"- **Specialty:** {expert.specialty}")
        lines.append(f"- **Perspective:** {expert.perspective}")
        if expert.archetype:
            lines.append(f"- **Archetype:** {expert.archetype}")
        if expert.model:
            lines.append(f"- **Model:** {expert.model}")
        lines.append("")

    for debate_round in rounds:
        lines.append(f"## Round {debate_round.round_number}\n")
        for msg in debate_round.messages:
            lines.append(f"### {msg.expert.name} ({msg.expert.specialty})\n")
            lines.append(msg.content)
            lines.append("")
        if debate_round.moderator_note:
            lines.append(f"**Moderator note:** {debate_round.moderator_note}\n")

    lines.append("## Final Synthesis\n")
    status = "Consensus reached" if result.reached else "Max rounds — forced synthesis"
    lines.append(f"**Status:** {status}  |  **Rounds:** {result.rounds_taken}\n")
    lines.append(result.synthesis)
    lines.append("")

    if result.expert_positions:
        agreed = [(e, s, a) for e, s, a in result.expert_positions if a]
        dissented = [(e, s, a) for e, s, a in result.expert_positions if not a]
        lines.append("## Expert Positions\n")
        for expert, summary, agreed_flag in agreed + dissented:
            label = "agreed" if agreed_flag else "dissent"
            lines.append(f"### {expert.name} — {label}\n")
            lines.append(summary)
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[dim]Debate saved to [/dim][bright_blue]{path}[/bright_blue]")


def _export_debug(question: str, debug_lines: list[str], console: Console) -> None:
    slug = _make_slug(question)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    path = exports_dir / f"{timestamp}_{slug}_debug.md"
    path.write_text("\n".join(debug_lines), encoding="utf-8")
    console.print(f"[dim]Debug log saved to [/dim][bright_blue]{path}[/bright_blue]")


def main() -> None:
    args = parse_args()
    console = Console()

    config = load_config(args.config)

    if args.tier:
        apply_tier(config, args.tier)
    if args.roles:
        config.roles_mode = args.roles
    if args.no_stream:
        config.stream = False
    if args.show_usage:
        config.show_usage = True
    if args.debug:
        config.debug = True

    console.print("[dim]Checking model availability…[/dim]")
    failures = check_models(config)
    if failures:
        console.print("[red]The following models are unavailable:[/red]")
        for model, reason in failures:
            console.print(f"  [red]• {model}[/red]: {reason}")
        console.print("[red]Aborting. Fix your config.toml or check your API key.[/red]")
        sys.exit(1)
    console.print("[dim]All models available.[/dim]")

    question = resolve_question(args, console)
    if not question:
        console.print("[red]Error: no question provided.[/red]")
        sys.exit(1)

    try:
        run(question, config, console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
