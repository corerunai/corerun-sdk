"""
Evaluation CLI commands
"""

import json
from typing import List, Optional

import typer
from rich.console import Console

from corerun.cli import output
from rich.table import Table

console = output.console
app = typer.Typer(help="LLM evaluation commands")

# Sub-groups
datasets_app = typer.Typer(help="Evaluation dataset management")
runs_app = typer.Typer(help="Evaluation run management")

app.add_typer(datasets_app, name="datasets")
app.add_typer(runs_app, name="runs")


def _init_client():
    try:
        from corerun import init
        return init()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)


def _run_status_style(status: str) -> str:
    styles = {
        "pending": "yellow",
        "running": "blue",
        "completed": "green",
        "failed": "red",
    }
    return styles.get(status, "white")


# ---------------------------------------------------------------------------
# Dataset commands
# ---------------------------------------------------------------------------

@datasets_app.command("list")
def list_datasets(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List evaluation datasets.

    Example:
        corerun evaluate datasets list
    """
    _init_client()

    import corerun.evaluations as ev

    try:
        datasets = ev.list_datasets(workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit([d.model_dump(mode="json") for d in datasets])
        return

    if not datasets:
        console.print("No evaluation datasets found")
        return

    table = Table(title="Evaluation Datasets")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name", style="cyan")
    table.add_column("Examples", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Created")

    for ds in datasets:
        table.add_row(
            ds.id,
            ds.name,
            str(ds.example_count),
            str(len(ds.columns)),
            ds.created_at.strftime("%Y-%m-%d %H:%M") if ds.created_at else "-",
        )

    console.print(table)


@datasets_app.command("get")
def get_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    show_examples: bool = typer.Option(False, "--examples", help="Show examples"),
):
    """
    Get evaluation dataset details.

    Example:
        corerun evaluate datasets get abc123
        corerun evaluate datasets get abc123 --examples
    """
    _init_client()

    import corerun.evaluations as ev

    try:
        ds = ev.get_dataset(dataset_id, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(ds.model_dump(mode="json"))
        return

    console.print(f"[bold]Dataset: {ds.name}[/bold]")
    console.print(f"  ID: {ds.id}")
    console.print(f"  Examples: {ds.example_count}")
    if ds.description:
        console.print(f"  Description: {ds.description}")
    console.print(f"  Schema: {', '.join(c.name for c in ds.columns)}")
    if ds.created_at:
        console.print(f"  Created: {ds.created_at}")

    if show_examples and ds.example_count > 0:
        # Fetch full dataset with examples
        try:
            full = ev.get_dataset(dataset_id, workspace=workspace)
            console.print(f"\nExamples (showing up to 5):")
            # The API returns examples in GetDataset — handled via JSON output for now
            console.print("[dim]Use --json to see examples[/dim]")
        except Exception:
            pass


@datasets_app.command("create")
def create_dataset(
    name: str = typer.Option(..., "--name", "-n", help="Dataset name"),
    schema_file: Optional[str] = typer.Option(None, "--schema", "-s", help="Path to JSON schema file"),
    examples_file: Optional[str] = typer.Option(None, "--examples", "-e", help="Path to JSON examples file"),
    description: Optional[str] = typer.Option(None, "--description", help="Description"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Create an evaluation dataset.

    Schema file should be a JSON array of column definitions:
    [{"name": "input", "type": "string", "required": true}, ...]

    Examples file should be a JSON array of row objects:
    [{"input": "...", "expected_output": "..."}, ...]

    Example:
        corerun evaluate datasets create --name qa-bench \\
            --schema schema.json --examples data.json
    """
    _init_client()

    import corerun.evaluations as ev
    from pathlib import Path

    if not schema_file:
        console.print("[red]Error:[/red] --schema is required")
        raise typer.Exit(1)

    schema_path = Path(schema_file)
    if not schema_path.exists():
        console.print(f"[red]Error:[/red] Schema file not found: {schema_file}")
        raise typer.Exit(1)

    try:
        schema = json.loads(schema_path.read_text())
    except Exception as e:
        console.print(f"[red]Error:[/red] Invalid schema JSON: {e}")
        raise typer.Exit(1)

    examples = None
    if examples_file:
        ex_path = Path(examples_file)
        if not ex_path.exists():
            console.print(f"[red]Error:[/red] Examples file not found: {examples_file}")
            raise typer.Exit(1)
        try:
            examples = json.loads(ex_path.read_text())
        except Exception as e:
            console.print(f"[red]Error:[/red] Invalid examples JSON: {e}")
            raise typer.Exit(1)

    try:
        ds = ev.create_dataset(
            name=name,
            schema=schema,
            description=description,
            examples=examples,
            workspace=workspace,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Created dataset '{ds.name}'")
    console.print(f"  ID: {ds.id}")
    console.print(f"  Examples: {ds.example_count}")
    console.print(f"  Columns: {', '.join(c.name for c in ds.columns)}")


@datasets_app.command("delete")
def delete_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Delete an evaluation dataset.

    Example:
        corerun evaluate datasets delete abc123 --force
    """
    _init_client()

    import corerun.evaluations as ev

    if not force:
        confirm = typer.confirm(f"Delete dataset '{dataset_id}'?")
        if not confirm:
            console.print("Cancelled")
            raise typer.Exit(0)

    try:
        ev.delete_dataset(dataset_id, workspace=workspace)
        console.print(f"[green]✓[/green] Deleted dataset '{dataset_id}'")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Run commands
# ---------------------------------------------------------------------------

@runs_app.command("list")
def list_runs(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List evaluation runs.

    Example:
        corerun evaluate runs list
        corerun evaluate runs list --status completed
    """
    _init_client()

    import corerun.evaluations as ev

    try:
        runs = ev.list_runs(status=status, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit([r.model_dump(mode="json") for r in runs])
        return

    if not runs:
        console.print("No evaluation runs found")
        return

    table = Table(title="Evaluation Runs")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name", style="cyan")
    table.add_column("Dataset")
    table.add_column("Scorers")
    table.add_column("Status")
    table.add_column("Progress", justify="right")
    table.add_column("Created")

    for run in runs:
        status_text = f"[{_run_status_style(run.status)}]{run.status}[/]"
        table.add_row(
            run.id,
            run.name,
            (run.dataset_name or run.dataset_id)[:20],
            ", ".join(run.scorers[:3]) + ("..." if len(run.scorers) > 3 else ""),
            status_text,
            f"{run.progress}%",
            run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "-",
        )

    console.print(table)


@runs_app.command("get")
def get_run(
    run_id: str = typer.Argument(..., help="Run ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    show_results: bool = typer.Option(False, "--results", help="Show aggregate score results"),
):
    """
    Get evaluation run details.

    Example:
        corerun evaluate runs get abc123
        corerun evaluate runs get abc123 --results
    """
    _init_client()

    import corerun.evaluations as ev

    try:
        run = ev.get_run(run_id, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(run.model_dump(mode="json"))
        return

    status_text = f"[{_run_status_style(run.status)}]{run.status}[/]"
    console.print(f"[bold]Evaluation Run: {run.name}[/bold]")
    console.print(f"  ID: {run.id}")
    console.print(f"  Status: {status_text}  Progress: {run.progress}%")
    console.print(f"  Dataset: {run.dataset_name or run.dataset_id}")
    console.print(f"  Model: {run.model_config.model} ({run.model_config.type})")
    console.print(f"  Scorers: {', '.join(run.scorers)}")
    if run.error:
        console.print(f"  Error: [red]{run.error}[/red]")
    if run.created_at:
        console.print(f"  Created: {run.created_at}")
    if run.completed_at:
        console.print(f"  Completed: {run.completed_at}")

    if show_results and run.results:
        agg = run.results.get("aggregate_scores", {})
        if agg:
            console.print("\n[bold]Aggregate Scores:[/bold]")
            for scorer, score in agg.items():
                console.print(f"  {scorer}: {score:.4f}")
        completed = run.results.get("completed", 0)
        total = run.results.get("total_examples", 0)
        console.print(f"\n  Completed: {completed}/{total} examples")


@runs_app.command("create")
def create_run(
    name: str = typer.Option(..., "--name", "-n", help="Run name"),
    dataset_id: str = typer.Option(..., "--dataset", "-d", help="Evaluation dataset ID"),
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    endpoint_id: Optional[str] = typer.Option(None, "--endpoint", help="corerun inference server ID"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Model API base URL"),
    api_key_val: Optional[str] = typer.Option(None, "--api-key", help="Model API key"),
    scorers: str = typer.Option(..., "--scorers", "-s", help="Comma-separated scorer IDs"),
    description: Optional[str] = typer.Option(None, "--description", help="Description"),
    compute: Optional[str] = typer.Option(None, "--compute", "-c", help="Auto-start on this compute target"),
    wait_for_completion: bool = typer.Option(False, "--wait", help="Wait for completion (requires --compute)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Create an evaluation run.

    Example:
        # With corerun inference endpoint
        corerun evaluate runs create --name my-eval --dataset abc123 \\
            --model mistral-7b --endpoint <server-id> \\
            --scorers correctness,fluency --compute dgx-cluster --wait

        # With external API (OpenAI, etc.)
        corerun evaluate runs create --name gpt-eval --dataset abc123 \\
            --model gpt-4o-mini --base-url https://api.openai.com/v1 \\
            --api-key sk-xxx --scorers correctness,fluency
    """
    _init_client()

    import corerun.evaluations as ev

    # Build model config
    if endpoint_id:
        model_config = {
            "type": "inference_endpoint",
            "endpoint_id": endpoint_id,
            "model": model,
        }
    elif base_url:
        model_config = {
            "type": "external_api",
            "base_url": base_url,
            "api_key": api_key_val or "",
            "model": model,
        }
    else:
        console.print("[red]Error:[/red] Provide either --endpoint or --base-url")
        raise typer.Exit(1)

    scorer_list = [s.strip() for s in scorers.split(",")]

    try:
        run = ev.create_run(
            name=name,
            dataset_id=dataset_id,
            model_config=model_config,
            scorers=scorer_list,
            description=description,
            workspace=workspace,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Created evaluation run '{run.name}'")
    console.print(f"  ID: {run.id}")
    console.print(f"  Status: [{_run_status_style(run.status)}]{run.status}[/]")

    if compute:
        console.print(f"\nStarting run on {compute}...")
        try:
            run = ev.start_run(run.id, compute_name=compute, workspace=workspace)
            console.print(f"  Status: [{_run_status_style(run.status)}]{run.status}[/]")

            if wait_for_completion:
                console.print("\nWaiting for completion...")
                try:
                    def on_update(r):
                        console.print(f"  Status: [{_run_status_style(r.status)}]{r.status}[/]  Progress: {r.progress}%")

                    run = ev.wait_run(run.id, callback=on_update, workspace=workspace)
                    console.print(f"\n[green]✓[/green] Evaluation completed")
                    console.print(f"  Status: [{_run_status_style(run.status)}]{run.status}[/]")
                    if run.results:
                        agg = run.results.get("aggregate_scores", {})
                        if agg:
                            console.print("\n[bold]Scores:[/bold]")
                            for scorer, score in agg.items():
                                console.print(f"  {scorer}: {score:.4f}")
                except KeyboardInterrupt:
                    console.print("\nInterrupted (run continues)")
        except Exception as e:
            console.print(f"[red]Error starting run:[/red] {e}")
            raise typer.Exit(1)
    else:
        console.print(f"\n  Use 'corerun evaluate runs start {run.id} --compute <name>' to start")


@runs_app.command("start")
def start_run(
    run_id: str = typer.Argument(..., help="Run ID"),
    compute: str = typer.Option(..., "--compute", "-c", help="Compute target name"),
    internal_network: bool = typer.Option(False, "--internal", help="Use cluster-internal inference URLs"),
    wait_for_completion: bool = typer.Option(False, "--wait", help="Wait for completion"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Start a pending evaluation run.

    Example:
        corerun evaluate runs start abc123 --compute dgx-cluster
        corerun evaluate runs start abc123 --compute dgx-cluster --wait
    """
    _init_client()

    import corerun.evaluations as ev

    try:
        run = ev.start_run(run_id, compute_name=compute,
                           use_internal_network=internal_network,
                           workspace=workspace)
        console.print(f"[green]✓[/green] Started evaluation run '{run.name}'")
        console.print(f"  Status: [{_run_status_style(run.status)}]{run.status}[/]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if wait_for_completion:
        console.print("\nWaiting for completion...")
        try:
            def on_update(r):
                console.print(f"  Status: [{_run_status_style(r.status)}]{r.status}[/]  Progress: {r.progress}%")

            run = ev.wait_run(run.id, callback=on_update, workspace=workspace)
            console.print(f"\n[green]✓[/green] Evaluation completed")
            if run.results:
                agg = run.results.get("aggregate_scores", {})
                if agg:
                    console.print("\n[bold]Scores:[/bold]")
                    for scorer, score in agg.items():
                        console.print(f"  {scorer}: {score:.4f}")
        except KeyboardInterrupt:
            console.print("\nInterrupted (run continues)")


@runs_app.command("wait")
def wait_for_run(
    run_id: str = typer.Argument(..., help="Run ID"),
    timeout: Optional[int] = typer.Option(None, "--timeout", "-t", help="Timeout in seconds"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Wait for an evaluation run to complete.

    Example:
        corerun evaluate runs wait abc123
        corerun evaluate runs wait abc123 --timeout 1800
    """
    _init_client()

    import corerun.evaluations as ev

    console.print(f"Waiting for evaluation run {run_id}...")

    try:
        def on_update(run):
            console.print(f"  Status: [{_run_status_style(run.status)}]{run.status}[/]  Progress: {run.progress}%")

        run = ev.wait_run(run_id, timeout=timeout, callback=on_update, workspace=workspace)
        console.print(f"\n[green]✓[/green] Completed")
        console.print(f"  Status: [{_run_status_style(run.status)}]{run.status}[/]")
        if run.results:
            agg = run.results.get("aggregate_scores", {})
            if agg:
                console.print("\n[bold]Aggregate Scores:[/bold]")
                for scorer, score in agg.items():
                    console.print(f"  {scorer}: {score:.4f}")
        if run.error:
            console.print(f"  Error: {run.error}")
    except KeyboardInterrupt:
        console.print("\nInterrupted (run continues)")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@runs_app.command("delete")
def delete_run(
    run_id: str = typer.Argument(..., help="Run ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Delete an evaluation run.

    Example:
        corerun evaluate runs delete abc123 --force
    """
    _init_client()

    import corerun.evaluations as ev

    if not force:
        confirm = typer.confirm(f"Delete evaluation run '{run_id}'?")
        if not confirm:
            console.print("Cancelled")
            raise typer.Exit(0)

    try:
        ev.delete_run(run_id, workspace=workspace)
        console.print(f"[green]✓[/green] Deleted evaluation run '{run_id}'")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Scorers command (top-level)
# ---------------------------------------------------------------------------

@app.command("scorers")
def list_scorers(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List available evaluation scorers.

    Example:
        corerun evaluate scorers
    """
    _init_client()

    import corerun.evaluations as ev

    try:
        scorers = ev.list_scorers(workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit([s.model_dump() for s in scorers])
        return

    table = Table(title="Available Scorers")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Type", style="dim")
    table.add_column("Description")

    for s in scorers:
        table.add_row(s.id, s.name, s.type, s.description)

    console.print(table)
