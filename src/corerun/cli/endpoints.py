"""
Model endpoint CLI commands.

An endpoint is the address an application calls. These commands create one, put
models behind it — yours or a provider's — and then send a real request through
it, which is the only way to know the whole path works.
"""

import sys
from typing import Optional

import typer
from rich.table import Table

from corerun.cli import output

console = output.console
app = typer.Typer(help="Model endpoint commands")


def _init_client():
    try:
        from corerun import init

        return init()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)


def _kind_style(kind: str) -> str:
    return "cyan" if kind == "external" else "magenta"


def _state_style(state: str) -> str:
    return {
        "running": "green",
        "available": "green",
        "deploying": "yellow",
        "pending": "yellow",
        "failed": "red",
        "stopped": "dim",
    }.get(state, "white")


@app.command("list")
def list_endpoints(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """List the endpoints in this workspace."""
    _init_client()
    import corerun.endpoints as endpoints

    items = endpoints.list(workspace=workspace)

    def render():
        if not items:
            console.print(
                "No endpoints yet. [cyan]corerun endpoints create <name>[/cyan] makes one."
            )
            return
        table = Table(title="Endpoints")
        table.add_column("Name", style="bold")
        table.add_column("Address")
        table.add_column("Models")
        table.add_column("Behind it", justify="right")
        for endpoint in items:
            table.add_row(
                endpoint.name,
                endpoint.url,
                ", ".join(endpoint.models) or "[dim]nothing serving[/dim]",
                str(endpoint.instances or len(endpoint.published)),
            )
        console.print(table)

    output.emit([e.model_dump() for e in items], render)


@app.command("show")
def show(
    name: str = typer.Argument(..., help="Endpoint name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    reveal: bool = typer.Option(False, "--reveal", help="Print the keys in full"),
):
    """Show one endpoint: its address, its keys, and what answers behind it."""
    _init_client()
    import corerun.endpoints as endpoints

    endpoint = endpoints.get(name, workspace=workspace)

    def mask(key: str) -> str:
        if not key:
            return "[dim]not generated[/dim]"
        return key if reveal else f"{key[:12]}{'•' * 12}"

    def render():
        console.print(f"\n[bold]{endpoint.name}[/bold]")
        if endpoint.description:
            console.print(f"[dim]{endpoint.description}[/dim]")
        console.print(f"\n  URL       {endpoint.url}")
        console.print(f"  Key 1     {mask(endpoint.api_key)}")
        console.print(f"  Key 2     {mask(endpoint.api_key_secondary)}")

        if not endpoint.published:
            console.print(
                "\n  [dim]Nothing published yet. Deploy a model into it, or"
                " add an external one.[/dim]\n"
            )
            return

        table = Table(title="Published models")
        table.add_column("Model", style="bold")
        table.add_column("Runs on")
        table.add_column("Accelerator")
        table.add_column("State")
        table.add_column("Upstream")
        for model in endpoint.published:
            table.add_row(
                model.model,
                f"[{_kind_style(model.kind)}]{model.where}[/]",
                # Blank for an external model: what somebody else's API runs
                # on is not something we can honestly report.
                model.accelerator or "",
                f"[{_state_style(model.state)}]{model.state}[/]",
                model.upstream_name if model.upstream_name != model.model else "",
            )
        console.print(table)

    output.emit(endpoint.model_dump(), render)


@app.command("create")
def create(
    name: str = typer.Argument(..., help="Name — it becomes part of the URL"),
    description: str = typer.Option("", "--description", "-d"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """
    Create an empty endpoint.

    It has a URL and a key straight away, so it can be handed out before
    anything is serving behind it.
    """
    _init_client()
    import corerun.endpoints as endpoints

    endpoint = endpoints.create(name, description=description, workspace=workspace)

    def render():
        console.print(f"[green]Created[/green] {endpoint.name}")
        console.print(f"  URL  {endpoint.url}")
        console.print(f"  Key  {endpoint.api_key}")
        console.print(
            "\n[dim]Nothing serves it yet. Deploy a model into it, or:[/dim]\n"
            f"  corerun endpoints add-upstream {endpoint.name} --model gpt-4o \\\n"
            "      --base-url https://api.openai.com/v1 --api-key sk-..."
        )

    output.emit(endpoint.model_dump(), render)


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Endpoint name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation"),
):
    """
    Delete an endpoint and the upstreams published on it.

    Refused while corerun is still running a deployment behind it — removing a
    name should not be how a running model is taken down. Upstreams go with it,
    since those are entries rather than workloads.
    """
    _init_client()
    import corerun.endpoints as endpoints

    if not yes and not output.json_mode():
        typer.confirm(f"Delete the endpoint '{name}'?", abort=True)

    result = endpoints.delete(name, workspace=workspace)
    output.emit(result, lambda: console.print(f"[green]Deleted[/green] {name}"))


@app.command("add-upstream")
def add_upstream(
    name: str = typer.Argument(..., help="Endpoint to publish on"),
    model: str = typer.Option(..., "--model", "-m", help="Name callers will ask for"),
    base_url: str = typer.Option(..., "--base-url", help="OpenAI-compatible root URL"),
    api_key: str = typer.Option("", "--api-key", help="The provider's credential"),
    upstream_name: str = typer.Option(
        "", "--as", help="What the provider is asked for, if it differs"
    ),
    provider: str = typer.Option("", "--provider", help="For display: openai, anthropic…"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """
    Publish a model corerun does not run.

    Use --as to publish one name onto another: an endpoint can offer
    "chat-large" and send it to gpt-4o, so what is behind the name can change
    without anybody's code changing.
    """
    _init_client()
    import corerun.endpoints as endpoints

    result = endpoints.add_upstream(
        name,
        model=model,
        base_url=base_url,
        api_key=api_key,
        upstream_name=upstream_name,
        provider=provider,
        workspace=workspace,
    )

    def render():
        console.print(f"[green]Published[/green] {model} on {name}")
        if upstream_name:
            console.print(f"  callers ask for {model} → sent upstream as {upstream_name}")
        console.print(f"  {base_url}")

    output.emit(result, render)


@app.command("remove-upstream")
def remove_upstream(
    name: str = typer.Argument(..., help="Endpoint name"),
    model: str = typer.Argument(..., help="The published name to stop serving"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """Stop publishing an upstream."""
    _init_client()
    import corerun.endpoints as endpoints

    result = endpoints.remove_upstream(name, model, workspace=workspace)
    output.emit(result, lambda: console.print(f"[green]Removed[/green] {model} from {name}"))


@app.command("rotate-key")
def rotate_key(
    name: str = typer.Argument(..., help="Endpoint name"),
    key: str = typer.Option("secondary", "--key", help="primary or secondary"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """
    Replace one of the two keys.

    Both work at once. Regenerate the one nobody uses, move callers onto it,
    then regenerate the other — that way nothing is ever without a key.
    """
    _init_client()
    import corerun.endpoints as endpoints

    result = endpoints.rotate_key(name, key=key, workspace=workspace)

    def render():
        console.print(f"[green]New {key} key[/green] for {name}")
        console.print(f"  {result.get('value', '')}")
        console.print("[dim]Move callers onto it, then rotate the other one.[/dim]")

    output.emit(result, render)


@app.command("models")
def models(
    name: str = typer.Argument(..., help="Endpoint name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """
    Ask the endpoint what it serves.

    Goes through the endpoint itself rather than the management API, so the
    answer is what a caller would actually get.
    """
    _init_client()
    import corerun.endpoints as endpoints

    try:
        served = endpoints.models(name, workspace=workspace)
    except Exception as e:  # noqa: BLE001 — the reason matters more than the type
        output.fail(f"The endpoint did not answer: {e}")
        return

    def render():
        if not served:
            console.print(f"{name} is reachable but serving nothing right now.")
            return
        for model in served:
            console.print(f"  {model}")

    output.emit({"models": served}, render)


@app.command("call")
def call(
    name: str = typer.Argument(..., help="Endpoint name"),
    prompt: str = typer.Argument(..., help="What to ask"),
    model: str = typer.Option("", "--model", "-m", help="Which model; omit if only one"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    stream_reply: bool = typer.Option(False, "--stream", help="Print as it arrives"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """
    Send a real request through the endpoint.

    This is the test that matters: the endpoint's own URL and key, the router
    choosing a model, the credential swapped for the upstream one. If this
    works, an application given the same two values works.
    """
    _init_client()
    import corerun.endpoints as endpoints

    try:
        if stream_reply:
            # Written straight to stdout and flushed, not through the console.
            # Rich holds output when stdout is not a terminal, which turns a
            # stream back into a wait-then-paste the moment anybody pipes this
            # into something -- and piping it is how you check that streaming
            # works at all. A token is also not markup: passing one through a
            # renderer that reads [brackets] would eat part of the answer.
            # A reasoning model produces its thinking first, and can spend a
            # whole budget on it without ever reaching an answer. Shown, or
            # the command looks hung for a minute and then prints nothing --
            # which is indistinguishable from a gateway that is buffering.
            #
            # On stderr, because it is progress and not the reply: piping the
            # command still gets the answer alone.
            thinking_shown = False

            def show_thinking(piece: str) -> None:
                nonlocal thinking_shown
                if not thinking_shown:
                    sys.stderr.write("thinking: ")
                    thinking_shown = True
                sys.stderr.write(piece)
                sys.stderr.flush()

            for piece in endpoints.stream(
                name, model, prompt, max_tokens=max_tokens,
                workspace=workspace, on_reasoning=show_thinking,
            ):
                if thinking_shown:
                    sys.stderr.write("\n\n")
                    sys.stderr.flush()
                    thinking_shown = False
                sys.stdout.write(piece)
                sys.stdout.flush()
            if thinking_shown:
                sys.stderr.write("\n")
                sys.stderr.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        reply = endpoints.complete(
            name, model, prompt, max_tokens=max_tokens, workspace=workspace
        )
    except Exception as e:  # noqa: BLE001
        output.fail(f"The call failed: {e}")
        return

    output.emit({"reply": reply}, lambda: console.print(reply))
