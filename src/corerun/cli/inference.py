"""
Inference server CLI commands
"""

import json as _json
from typing import Optional, List

import typer
from rich.console import Console

from corerun.cli import output, resolve
from rich.table import Table

console = output.console
app = typer.Typer(help="Inference server commands")


def _init_client():
    """Initialize client, handling errors gracefully"""
    try:
        from corerun import init
        return init()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)


def _resolve(reference: str, workspace) -> str:
    """A name, a full id, or the shortened id the list prints."""
    import corerun.inference as inference

    return resolve.by_name_or_id(
        reference, lambda: inference.list(workspace=workspace), "inference server"
    )


def _status_style(status: str) -> str:
    styles = {
        "pending": "yellow",
        "deploying": "yellow",
        "starting": "yellow",
        "running": "green",
        "stopped": "dim",
        "failed": "red",
    }
    return styles.get(status, "white")


@app.command("list")
def list_servers(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List inference servers.

    Example:
        corerun inference list
        corerun inference list --status running
    """
    _init_client()

    import corerun.inference as inference

    try:
        servers = inference.list(status=status, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit([s.model_dump(mode="json") for s in servers])
        return

    if not servers:
        console.print("No inference servers found")
        return

    table = Table(title="Inference Servers")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Model")
    table.add_column("Status", no_wrap=True)
    table.add_column("GPU", justify="right")
    table.add_column("Replicas", justify="right")
    table.add_column("ID", style="dim", no_wrap=True)

    for s in servers:
        model = s.model_id or "-"
        if len(model) > 30:
            model = model[:30] + "..."
        table.add_row(
            s.name,
            s.server_type,
            model,
            f"[{_status_style(s.status)}]{s.status}[/]",
            str(int(s.gpu)) if s.gpu else "-",
            str(s.replicas),
            s.id,
        )

    console.print(table)


@app.command("get")
def get_server(
    server_id: str = typer.Argument(..., metavar="SERVER", help="Server name or ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Show an inference server's details.

    Example:
        corerun inference get abc123
    """
    _init_client()

    import corerun.inference as inference

    resolved = _resolve(server_id, workspace)

    try:
        s = inference.get(resolved, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(s.model_dump(mode="json"))
        return

    console.print(f"\n[bold cyan]{s.name}[/]  [dim]{s.id}[/]")
    console.print(f"  Status:   [{_status_style(s.status)}]{s.status}[/]")
    console.print(f"  Type:     {s.server_type}")
    console.print(f"  Model:    {s.model_id} [dim]({s.model_source})[/]")
    console.print(f"  Compute:  {s.compute_name}")
    console.print(f"  GPU:      {s.gpu}")
    console.print(f"  Replicas: {s.replicas} [dim](min {s.min_replicas}, max {s.max_replicas})[/]")

    if s.openai_base_url:
        console.print(f"  OpenAI base URL: {s.openai_base_url}")
    if s.error:
        console.print(f"  [red]Error:[/red] {s.error}")

    if s.lora_modules:
        table = Table(title="LoRA Modules")
        table.add_column("Name", style="cyan")
        table.add_column("Source")
        for m in s.lora_modules:
            table.add_row(m.name, m.source or "-")
        console.print(table)


@app.command("deploy")
def deploy_server(
    name: str = typer.Option(..., "--name", "-n", help="Server name"),
    model: str = typer.Option(
        ..., "--model", "-m", help="Model ID (HuggingFace ID, registry name, or path)"
    ),
    compute: str = typer.Option(..., "--compute", "-c", help="Compute target name"),
    server_type: str = typer.Option(
        "vllm", "--type", "-t", help="Server engine: vllm, sglang or ollama"
    ),
    model_source: str = typer.Option(
        "huggingface", "--source",
        help="Where the weights come from: huggingface, mlflow, registry, or path "
             "for a directory already on the machine",
    ),
    endpoint: Optional[str] = typer.Option(
        None, "--endpoint", "-e",
        help="Publish behind this endpoint; defaults to a new one named after the deployment",
    ),
    served_name: Optional[List[str]] = typer.Option(
        None, "--served-name",
        help="What callers ask for; repeat for several. Needed for --source path, "
             "which would otherwise publish the directory as the model name",
    ),
    arg: Optional[List[str]] = typer.Option(
        None, "--arg",
        help="Engine flag passed through as given; repeat for each. "
             "e.g. --arg --kv-cache-dtype --arg fp8",
    ),
    model_version: Optional[str] = typer.Option(
        None, "--model-version", help="Version, for registry models"
    ),
    image: Optional[str] = typer.Option(
        None, "--image", help="Container image (defaults per server type)"
    ),
    gpu: float = typer.Option(1, "--gpu", "-g", help="GPUs per replica"),
    min_replicas: int = typer.Option(1, "--min-replicas", help="Minimum replicas"),
    max_replicas: int = typer.Option(1, "--max-replicas", help="Maximum replicas"),
    max_model_len: Optional[int] = typer.Option(
        None, "--max-model-len", help="Maximum context length (vLLM)"
    ),
    tensor_parallel: Optional[int] = typer.Option(
        None, "--tensor-parallel", help="Tensor parallel size (vLLM)"
    ),
    quantization: Optional[str] = typer.Option(
        None, "--quantization", "-q", help="Quantization: awq, gptq, fp8, ..."
    ),
    gpu_memory_util: Optional[float] = typer.Option(
        None, "--gpu-memory-util", help="GPU memory utilization, 0.0-1.0 (vLLM)"
    ),
    enforce_eager: bool = typer.Option(False, "--enforce-eager", help="Disable CUDA graphs (vLLM)"),
    enable_tracing: bool = typer.Option(False, "--tracing", help="Enable MLflow tracing"),
    wait: bool = typer.Option(False, "--wait", help="Wait until the server is running"),
    timeout: int = typer.Option(600, "--timeout", help="Seconds to wait when --wait is set"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Deploy an inference server.

    This allocates GPUs and counts against the workspace quota.

    Example:
        corerun inference deploy --name mistral --model mistralai/Mistral-7B \\
            --compute dgx --gpu 1
    """
    _init_client()

    import corerun.inference as inference

    try:
        server = inference.deploy(
            name=name,
            model_id=model,
            compute_name=compute,
            server_type=server_type,
            model_source=model_source,
            model_version=model_version,
            image=image,
            gpu=gpu,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            max_model_len=max_model_len,
            tensor_parallel=tensor_parallel,
            quantization=quantization,
            gpu_memory_util=gpu_memory_util,
            enforce_eager=enforce_eager,
            enable_tracing=enable_tracing,
            endpoint=endpoint,
            extra_args=list(arg) if arg else None,
            served_names=list(served_name) if served_name else None,
            wait=wait,
            timeout=timeout,
            workspace=workspace,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Deployed[/green] {server.name} [dim]({server.id})[/]")
    console.print(f"  Status: [{_status_style(server.status)}]{server.status}[/]")
    if server.openai_base_url:
        console.print(f"  OpenAI base URL: {server.openai_base_url}")
    if not wait:
        console.print(f"\n[dim]Follow with: corerun inference get {server.id}[/]")


@app.command("scale")
def scale_server(
    server_id: str = typer.Argument(..., metavar="SERVER", help="Server name or ID"),
    min_replicas: int = typer.Option(..., "--min-replicas", help="Minimum replicas"),
    max_replicas: int = typer.Option(..., "--max-replicas", help="Maximum replicas"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Scale an inference server's replicas.

    Scaling up allocates more GPUs.

    Example:
        corerun inference scale abc123 --min-replicas 2 --max-replicas 4
    """
    _init_client()

    import corerun.inference as inference

    try:
        server = inference.scale(
            server_id,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            workspace=workspace,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"[green]Scaled[/green] {server.name} to "
        f"min {server.min_replicas}, max {server.max_replicas}"
    )


@app.command("stop")
def stop_server(
    server_id: str = typer.Argument(..., metavar="SERVER", help="Server name or ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Stop an inference server, releasing its GPUs.

    Example:
        corerun inference stop abc123
    """
    _init_client()

    import corerun.inference as inference

    resolved = _resolve(server_id, workspace)

    try:
        server = inference.stop(resolved, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Stopped[/green] {server.name} [dim]({server.status})[/]")


@app.command("start")
def start_server(
    server_id: str = typer.Argument(..., metavar="SERVER", help="Server name or ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Start a stopped inference server.

    Example:
        corerun inference start abc123
    """
    _init_client()

    import corerun.inference as inference

    resolved = _resolve(server_id, workspace)

    try:
        server = inference.start(resolved, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Started[/green] {server.name} [dim]({server.status})[/]")


@app.command("restart")
def restart_server(
    server_id: str = typer.Argument(..., metavar="SERVER", help="Server name or ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Restart an inference server.

    Example:
        corerun inference restart abc123
    """
    _init_client()

    import corerun.inference as inference

    resolved = _resolve(server_id, workspace)

    try:
        inference.restart(resolved, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[green]Restart requested[/green]")


@app.command("delete")
def delete_server(
    server_id: str = typer.Argument(..., metavar="SERVER", help="Server name or ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip the confirmation prompt"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Delete an inference server.

    Example:
        corerun inference delete abc123
    """
    _init_client()

    import corerun.inference as inference

    resolved = _resolve(server_id, workspace)

    if not force and not typer.confirm(f"Delete inference server {server_id}?"):
        raise typer.Abort()

    try:
        inference.delete(resolved, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Deleted[/green] {server_id}")


@app.command("wait")
def wait_server(
    server_id: str = typer.Argument(..., metavar="SERVER", help="Server name or ID"),
    timeout: int = typer.Option(600, "--timeout", help="Maximum seconds to wait"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Wait for an inference server to reach running.

    Example:
        corerun inference wait abc123
    """
    _init_client()

    import corerun.inference as inference

    resolved = _resolve(server_id, workspace)

    try:
        server = inference.wait_for_running(resolved, timeout=timeout, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Running[/green] {server.name}")
    if server.openai_base_url:
        console.print(f"  OpenAI base URL: {server.openai_base_url}")


@app.command("types")
def list_types(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List the server engines this platform supports.

    Example:
        corerun inference types
    """
    _init_client()

    import corerun.inference as inference

    try:
        items = inference.list_types(workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        output.emit(items)
        return

    if not items:
        console.print("No server types available")
        return

    table = Table(title="Inference Server Types")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for t in items:
        if isinstance(t, dict):
            table.add_row(str(t.get("name", "-")), str(t.get("description", "")))
        else:
            table.add_row(str(t), "")

    console.print(table)
