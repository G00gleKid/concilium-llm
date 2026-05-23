from __future__ import annotations

from rich.columns import Columns
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from .models import ConsensusResult, Expert

EXPERT_COLORS = ["cyan", "magenta", "yellow"]


def _expert_color(experts: list[Expert], expert: Expert) -> str:
    try:
        idx = next(i for i, e in enumerate(experts) if e.name == expert.name)
        return EXPERT_COLORS[idx % len(EXPERT_COLORS)]
    except StopIteration:
        return "white"


def render_header(console: Console) -> None:
    console.print()
    console.print(
        Panel(
            "[bold white]CONCILIUM[/bold white]\n[dim]Multi-Expert Debate System[/dim]",
            border_style="bright_blue",
            expand=False,
            padding=(1, 4),
        )
    )
    console.print()


def render_question(question: str, console: Console) -> None:
    console.print(
        Panel(
            f"[bold]{escape(question)}[/bold]",
            title="[bright_blue]Question[/bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print()


def render_experts(experts: list[Expert], console: Console) -> None:
    console.print(Rule("[bright_blue]Expert Panel[/bright_blue]", style="bright_blue"))
    console.print()

    panels = []
    for i, expert in enumerate(experts):
        color = EXPERT_COLORS[i % len(EXPERT_COLORS)]
        archetype_line = f"\n[dim]Archetype: {expert.archetype}[/dim]" if expert.archetype else ""
        model_line = f"\n[dim]Model: {escape(expert.model)}[/dim]" if expert.model else ""
        content = (
            f"[bold]{escape(expert.specialty)}[/bold]{archetype_line}{model_line}\n\n"
            f"[italic]{escape(expert.perspective)}[/italic]\n\n"
            f"[dim]{escape(expert.rationale)}[/dim]"
        )
        panels.append(
            Panel(
                content,
                title=f"[{color}]{escape(expert.name)}[/{color}]",
                border_style=color,
                expand=True,
            )
        )

    console.print(Columns(panels, equal=True, expand=True))
    console.print()


def render_round_header(round_num: int, console: Console) -> None:
    console.print()
    console.print(Rule(f"[dim]Round {round_num}[/dim]", style="dim"))
    console.print()


def render_expert_turn_start(
    expert: Expert,
    experts: list[Expert],
    console: Console,
) -> None:
    color = _expert_color(experts, expert)
    model_tag = f" [{escape(expert.model)}]" if expert.model else ""
    console.print(
        f"[{color}][bold]{escape(expert.name)}[/bold][/{color}]"
        f" [dim]({escape(expert.specialty)}){model_tag}[/dim]"
    )


def render_expert_turn_end(console: Console) -> None:
    console.print()


def render_expert_message(
    expert: Expert,
    experts: list[Expert],
    content: str,
    console: Console,
) -> None:
    color = _expert_color(experts, expert)
    model_tag = f" [{escape(expert.model)}]" if expert.model else ""
    title = (
        f"[{color}][bold]{escape(expert.name)}[/bold][/{color}]"
        f" [dim]({escape(expert.specialty)}){model_tag}[/dim]"
    )
    console.print(
        Panel(
            escape(content),
            title=title,
            border_style=color,
            padding=(1, 2),
        )
    )
    console.print()


def render_moderator_note(
    note: str,
    confidence: float,
    position_variance: float | None,
    console: Console,
) -> None:
    bar_filled = int(confidence * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)
    variance_line = (
        f"  |  Divergence: {position_variance:.3f}" if position_variance is not None else ""
    )
    console.print(
        Panel(
            f"[dim]{escape(note)}[/dim]\n\nConsensus confidence: [{bar}] {confidence:.0%}{variance_line}",
            title="[dim]Moderator[/dim]",
            border_style="dim",
        )
    )
    console.print()


def render_synthesis(result: ConsensusResult, console: Console) -> None:
    console.print()
    console.print(Rule("[bright_green]Final Synthesis[/bright_green]", style="bright_green"))
    console.print()

    status = "[bright_green]Consensus reached[/bright_green]" if result.reached else "[yellow]Max rounds reached — forced synthesis[/yellow]"
    console.print(f"Status: {status}  |  Rounds: {result.rounds_taken}")
    console.print()

    console.print(
        Panel(
            escape(result.synthesis),
            title="[bright_green]Weighted Conclusion[/bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        )
    )
    console.print()


def render_dissents(result: ConsensusResult, console: Console) -> None:
    if not result.expert_positions:
        return
    console.print(Rule("[yellow]Expert Positions[/yellow]", style="yellow"))
    console.print()

    agreed = [(e, s, a) for e, s, a in result.expert_positions if a]
    dissented = [(e, s, a) for e, s, a in result.expert_positions if not a]

    for expert, summary, agreed_flag in agreed + dissented:
        if agreed_flag:
            label = "[bright_green]— agreed[/bright_green]"
            border = "bright_green"
        else:
            label = "[yellow]— dissent[/yellow]"
            border = "yellow"
        console.print(
            Panel(
                escape(summary),
                title=f"[bold]{escape(expert.name)}[/bold] {label}",
                border_style=border,
            )
        )
        console.print()


def render_usage(result: ConsensusResult, console: Console) -> None:
    u = result.usage
    console.print(Rule("[dim]Token Usage[/dim]", style="dim"))
    console.print(
        f"[dim]Input: {u.get('input_tokens', 0):,}  |  "
        f"Output: {u.get('output_tokens', 0):,}[/dim]"
    )
    console.print()
