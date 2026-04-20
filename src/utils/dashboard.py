"""
Rich terminal dashboard for paper-trading mode.

Replaces the noisy per-component log blocks with a single clean snapshot
printed at each metrics interval.
"""

from __future__ import annotations

import time
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_console = Console(highlight=False)


def _progress_bar(fraction: float, width: int = 32) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    color = "red" if fraction > 0.80 else "yellow" if fraction > 0.50 else "green"
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"


def render_paper_dashboard(
    *,
    runtime_seconds: float,
    initial_balance: float,
    summary: dict,
    positions: list[dict],
    risk: dict,
    fills_total: int,
    fill_volume_usdc: float,
) -> None:
    """
    Print a full paper-trade dashboard to the terminal.

    Args:
        runtime_seconds:  seconds since bot started
        initial_balance:  starting USDC (e.g. 20_000)
        summary:          PaperOrderManager.summary() dict
        positions:        InventoryManager.all_open_positions() list
        risk:             RiskManager.status() dict
        fills_total:      total fills from MetricsCollector
        fill_volume_usdc: total fill volume from MetricsCollector
    """
    balance      = summary["balance_usdc"]
    locked       = summary["locked_usdc"]
    in_positions = max(0.0, initial_balance - balance)
    free         = max(0.0, balance - locked)
    realized_pnl   = summary["realized_pnl"]
    unrealized_pnl = summary.get("unrealized_pnl", 0.0)
    total_pnl      = realized_pnl + unrealized_pnl
    reward_est     = summary["reward_earned_usdc"]
    open_orders  = summary["open_orders"]
    open_yes     = summary["open_yes_orders"]
    open_no      = summary["open_no_orders"]

    total_committed  = in_positions + locked
    deployed_pct     = total_committed / initial_balance if initial_balance > 0 else 0.0

    # Runtime
    m, s = divmod(int(runtime_seconds), 60)
    h, m = divmod(m, 60)
    runtime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    pnl_color     = "green" if realized_pnl >= 0 else "red"
    unreal_color  = "green" if unrealized_pnl >= 0 else "red"
    total_color   = "green" if total_pnl >= 0 else "red"

    # ── Capital panel ─────────────────────────────────────────────────────────
    cap = Table(box=None, show_header=False, padding=(0, 1))
    cap.add_column(style="dim", min_width=20)
    cap.add_column(justify="right", min_width=12)
    cap.add_row("Starting capital",   f"[bold]${initial_balance:>10,.2f}[/bold]")
    cap.add_row("Cash balance",       f"${balance:>10,.2f}")
    cap.add_row("In open positions",  f"[cyan]${in_positions:>9,.2f}[/cyan]")
    cap.add_row("Reserved (orders)",  f"[yellow]${locked:>9,.2f}[/yellow]")
    cap.add_row("Free cash",          f"[green]${free:>10,.2f}[/green]")

    # ── Performance panel ─────────────────────────────────────────────────────
    perf = Table(box=None, show_header=False, padding=(0, 1))
    perf.add_column(style="dim", min_width=20)
    perf.add_column(justify="right", min_width=12)
    perf.add_row("Realized PnL",      f"[{pnl_color}]${realized_pnl:>9,.2f}[/{pnl_color}]")
    perf.add_row("Unrealized PnL",   f"[{unreal_color}]${unrealized_pnl:>9,.2f}[/{unreal_color}]")
    perf.add_row("Total PnL",        f"[bold {total_color}]${total_pnl:>9,.2f}[/bold {total_color}]")
    perf.add_row("Fill volume",       f"${fill_volume_usdc:>10,.2f}")
    perf.add_row("Total fills",       f"{fills_total:>12,}")
    perf.add_row(
        "Rewards (est. max)",
        f"[dim]${reward_est:>9,.2f}[/dim]",
    )

    # ── Positions table ───────────────────────────────────────────────────────
    pos_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        expand=True,
    )
    pos_table.add_column("Market",    style="dim",    min_width=13)
    pos_table.add_column("Token",     justify="center", min_width=6)
    pos_table.add_column("Shares",    justify="right", min_width=8)
    pos_table.add_column("Avg Cost",  justify="right", min_width=9)
    pos_table.add_column("Deployed",  justify="right", min_width=10)
    pos_table.add_column("% Capital", justify="right", min_width=9)

    for p in sorted(positions, key=lambda x: x["cost_basis"], reverse=True):
        cost = p["cost_basis"]
        pct  = cost / initial_balance * 100 if initial_balance else 0
        tok_style = "cyan" if p["token"] == "YES" else "magenta"
        pos_table.add_row(
            p["condition_id"][:10] + "…",
            f"[{tok_style}]{p['token']}[/{tok_style}]",
            f"{p['shares']:>8,.0f}",
            f"${p['avg_cost']:.3f}",
            f"[bold]${cost:>8,.2f}[/bold]",
            f"{pct:.1f}%",
        )

    if not positions:
        pos_table.add_row(
            "[dim]No open positions yet[/dim]", "", "", "", "", ""
        )

    # ── Orders bar ────────────────────────────────────────────────────────────
    bar = _progress_bar(deployed_pct)
    orders_text = Text.from_markup(
        f"{open_orders} open orders  "
        f"[dim]({open_yes} YES bids  ·  {open_no} NO bids)[/dim]\n"
        f"{bar}  [bold]{deployed_pct * 100:.0f}%[/bold] of capital committed"
    )

    # ── Risk line ─────────────────────────────────────────────────────────────
    cb_tripped  = risk.get("global_cb_tripped", False)
    drawdown    = risk.get("current_drawdown", 0.0)
    mkts_paused = risk.get("market_cbs_tripped", [])
    dd_color    = "red" if drawdown > 500 else "yellow" if drawdown > 100 else "green"
    cb_str      = "[red]⚠  TRIPPED[/red]" if cb_tripped else "[green]CLEAR[/green]"
    risk_parts  = [
        f"Drawdown: [{dd_color}]${drawdown:,.2f}[/{dd_color}]",
        f"Circuit breaker: {cb_str}",
    ]
    if mkts_paused:
        risk_parts.append(f"[yellow]Markets paused: {len(mkts_paused)}[/yellow]")
    risk_text = Text.from_markup("  ·  ".join(risk_parts))

    # ── Print ─────────────────────────────────────────────────────────────────
    _console.rule(
        f"[bold dim]{time.strftime('%H:%M:%S')}  ·  "
        f"Paper Trade  ·  {runtime_str}[/bold dim]"
    )
    _console.print(
        Columns([
            Panel(cap,  title="[bold]Capital[/bold]",     border_style="blue",   expand=True),
            Panel(perf, title="[bold]Performance[/bold]", border_style="blue",   expand=True),
        ])
    )
    _console.print(
        Panel(pos_table, title="[bold]Open Positions[/bold]", border_style="cyan", expand=True)
    )
    _console.print(
        Panel(orders_text, title="[bold]Orders[/bold]", border_style="yellow", expand=True)
    )
    _console.print(
        Panel(
            risk_text,
            title="[bold]Risk[/bold]",
            border_style="red" if cb_tripped else "dim",
            expand=True,
        )
    )
