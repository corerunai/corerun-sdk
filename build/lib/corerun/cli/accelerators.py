"""
What each accelerator generation can do.

Read-only, and the same table the platform deploys from: the KV cache dtypes
a card holds, the formats it has kernels for, the toolkit an image must be
built for, and the arguments a family adds to every model it serves.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from corerun.cli import output

console = output.console
app = typer.Typer(help="Accelerator generations and what they support")


def _init_client():
    try:
        from corerun import init

        return init()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def accelerators(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Print results as JSON, for piping into other tools"),
):
    """What each accelerator generation can do."""
    if json_output:
        output.set_json(True)

    if ctx.invoked_subcommand is None:
        list_families(json_output=json_output)


def _table(payload: dict, revision: str) -> Table:
    table = Table(title=f"Accelerators (table {revision})")
    table.add_column("Family", style="cyan", no_wrap=True)
    table.add_column("Cards")
    table.add_column("KV cache")
    table.add_column("Toolkit")
    table.add_column("Interconnect")

    for name in sorted(payload):
        caps = payload[name]
        table.add_row(
            name,
            caps.get("display_name", ""),
            ", ".join(caps.get("kv_cache_dtypes") or []) or "[dim]not claimed[/dim]",
            " ".join(x for x in [caps.get("runtime", ""), caps.get("min_runtime", "")] if x) or "-",
            caps.get("interconnect", "-"),
        )
    return table


@app.command("list")
def list_families(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List the accelerator generations this platform knows.

    Example:
        corerun accelerators list
    """
    _init_client()

    from corerun.config import get_client

    try:
        payload = get_client().get("/inference-servers/accelerators") or {}
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(payload)
        return

    families = payload.get("families") or {}
    if not families:
        console.print("No accelerators known")
        return

    console.print(_table(families, payload.get("revision", "?")))
    console.print(f"[dim]Read from {payload.get('origin', 'unknown')}[/dim]")


@app.command("show")
def show_accelerator(
    token: str = typer.Argument(..., metavar="FAMILY|CARD", help="e.g. hopper, h100, rtx pro 5000"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    What one accelerator is, and what it can do.

    A family name or the card itself, which is how a person knows what they
    have. Something the platform has never classified is reported as
    unclassified rather than refused: those machines are deployed to.

    Example:
        corerun accelerators show h100
        corerun accelerators show "rtx pro 5000"
    """
    _init_client()

    from corerun.config import get_client

    try:
        payload = get_client().get(f"/inference-servers/accelerators/{token}") or {}
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(payload)
        return

    if not payload.get("known"):
        console.print(f"[yellow]{token} is not an accelerator this platform has classified.[/yellow]")
        console.print("Deployments proceed with the platform's own defaults, unchanged.")
        return

    caps = payload.get("capabilities") or {}
    console.print(f"\n[bold cyan]{caps.get('display_name', token)}[/]  [dim]{payload.get('family')}[/]")
    if caps.get("compute_capability"):
        console.print(f"  Compute capability: {caps['compute_capability']}")
    if caps.get("runtime"):
        console.print(f"  Toolkit:            {caps['runtime']} {caps.get('min_runtime', '')}".rstrip())
    if caps.get("min_vllm"):
        console.print(f"  Needs vLLM:         {caps['min_vllm']} or newer")
    if caps.get("kv_cache_dtypes"):
        console.print(f"  KV cache:           {', '.join(caps['kv_cache_dtypes'])}")
    if caps.get("quantizations"):
        console.print(f"  Quantizations:      {', '.join(caps['quantizations'])}")
    if caps.get("interconnect"):
        console.print(f"  Interconnect:       {caps['interconnect']}")
    if caps.get("unified_memory"):
        console.print("  Memory:             unified with the CPU")
    if caps.get("args"):
        console.print(f"  Always launched with: {' '.join(caps['args'])}")
    if caps.get("note"):
        console.print(f"  [yellow]·[/yellow] {caps['note']}")
    if caps.get("source"):
        console.print(f"  [dim]{caps['source']}[/dim]")
