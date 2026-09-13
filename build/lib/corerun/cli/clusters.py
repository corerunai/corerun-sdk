"""
Cluster CLI commands
"""

import json as _json
from typing import Optional

import typer
from rich.console import Console

from corerun.cli import output
from rich.table import Table

console = output.console
app = typer.Typer(help="Cluster inspection commands")


def _init_client():
    """Initialize client, handling errors gracefully"""
    try:
        from corerun import init
        return init()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)


def _status_style(status: str) -> str:
    styles = {
        "ready": "green",
        "pending": "yellow",
        "error": "red",
    }
    return styles.get(status, "white")


@app.command("list")
def list_clusters(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List clusters available to the workspace.

    Example:
        corerun clusters list
    """
    _init_client()

    import corerun.clusters as clusters

    try:
        items = clusters.list(workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit([c.model_dump(mode="json") for c in items])
        return

    if not items:
        console.print("No clusters found")
        return

    table = Table(title="Clusters")
    table.add_column("Name", style="cyan")
    table.add_column("Backend")
    table.add_column("Scope")
    table.add_column("Status")
    table.add_column("Connector")
    table.add_column("Nodes", justify="right")
    table.add_column("GPUs", justify="right")

    for c in items:
        connector = "[green]connected[/]" if c.connector_connected else "[dim]offline[/]"
        nodes = str(c.resources.node_count) if c.resources else "-"
        if c.resources and c.resources.total_gpus:
            gpus = f"{c.resources.allocated_gpus}/{c.resources.total_gpus}"
        else:
            gpus = "-"
        table.add_row(
            c.name,
            c.backend or "-",
            c.scope or "-",
            f"[{_status_style(c.status)}]{c.status}[/]",
            connector,
            nodes,
            gpus,
        )

    console.print(table)


@app.command("get")
def get_cluster(
    name: str = typer.Argument(..., help="Cluster name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Show a cluster's details and live capacity.

    Example:
        corerun clusters get gb10dgx01
    """
    _init_client()

    import corerun.clusters as clusters

    try:
        c = clusters.get(name, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(c.model_dump(mode="json"))
        return

    console.print(f"\n[bold cyan]{c.name}[/]")
    console.print(f"  Status:       [{_status_style(c.status)}]{c.status}[/]")
    console.print(f"  Backend:      {c.backend or '-'}")
    console.print(f"  Namespace:    {c.namespace or '-'}")
    console.print(f"  Scope:        {c.scope or '-'}")
    console.print(f"  Architecture: {c.architecture or '-'}")
    console.print(f"  GPU strategy: {c.gpu_strategy or '-'}")
    console.print(f"  Connector:    {'connected' if c.connector_connected else 'offline'}")

    if not c.resources:
        console.print("\n[dim]No resource data — the connector has not reported in.[/]")
        return

    r = c.resources
    console.print("\n[bold]Capacity[/]")
    console.print(f"  Nodes:  {r.node_count}")
    console.print(f"  CPU:    {r.allocated_cpu}/{r.total_cpu}")
    console.print(f"  Memory: {r.allocated_memory_mb}/{r.total_memory_mb} MB")
    console.print(f"  GPUs:   {r.allocated_gpus}/{r.total_gpus} ({r.available_gpus} free)")

    if r.gpus:
        table = Table(title="GPUs")
        table.add_column("Vendor")
        table.add_column("Model")
        table.add_column("Count", justify="right")
        for g in r.gpus:
            table.add_row(g.vendor or "-", g.model or "-", str(g.count))
        console.print(table)

    if r.storage_classes:
        table = Table(title="Storage Classes")
        table.add_column("Name", style="cyan")
        table.add_column("Provisioner")
        table.add_column("Default")
        for sc in r.storage_classes:
            table.add_row(sc.name, sc.provisioner or "-", "yes" if sc.is_default else "")
        console.print(table)


@app.command("profiles")
def list_profiles(
    name: str = typer.Argument(..., help="Cluster name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List the pod profiles a cluster offers.

    Example:
        corerun clusters profiles gb10dgx01
    """
    _init_client()

    import corerun.clusters as clusters

    try:
        items = clusters.profiles(name, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(items)
        return

    if not items:
        console.print("No profiles defined")
        return

    table = Table(title=f"Profiles — {name}")
    table.add_column("Name", style="cyan")
    table.add_column("CPU", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("GPU", justify="right")

    for p in items:
        if not isinstance(p, dict):
            continue
        table.add_row(
            str(p.get("name", "-")),
            str(p.get("cpu_limit", p.get("cpu", "-"))),
            str(p.get("memory_limit", p.get("memory", "-"))),
            str(p.get("gpu", "-")),
        )

    console.print(table)
