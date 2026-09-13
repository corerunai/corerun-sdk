"""
Workspace selection.

Which workspace a command acts on is the single piece of context every other
command inherits, so it gets its own place rather than living inside auth.
"""

import typer
from rich.console import Console
from rich.table import Table

from corerun.cli import output
from corerun.config import get_config

console = output.console
app = typer.Typer(help="Workspace selection")


def _require_credentials():
    config = get_config()
    if not config.auth_token:
        console.print("[red]Error:[/red] not signed in. Run 'corerun login' first.")
        raise typer.Exit(1)
    return config


def fetch(api_url: str, api_key: str):
    """The workspaces this credential can act in."""
    import httpx

    response = httpx.get(
        api_url.rstrip("/") + "/workspaces",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json().get("workspaces", [])


def label(workspace: dict) -> str:
    """Name a workspace the way its owner would recognise it."""
    name = workspace.get("display_name") or workspace.get("slug") or workspace.get("id", "")
    tenant = workspace.get("tenant_slug")
    return f"{tenant}/{name}" if tenant else name


def resolve(workspaces, wanted: str):
    """Find a workspace by whatever the user typed.

    An ID, a slug, a display name, or "tenant/name" as it is printed back to
    them -- because the name shown is the name people will type, and demanding
    a UUID for something they have just seen listed by name is a poor trade for
    a few lines of matching.
    """
    wanted = wanted.strip()
    for workspace in workspaces:
        if workspace.get("id") == wanted:
            return workspace
    lowered = wanted.lower()
    for workspace in workspaces:
        candidates = {
            (workspace.get("slug") or "").lower(),
            (workspace.get("display_name") or "").lower(),
            label(workspace).lower(),
        }
        if lowered in candidates:
            return workspace
    return None


def choose(api_url: str, api_key: str, default: str = None):
    """List the caller's workspaces and ask which one to use.

    A single workspace is chosen without asking -- there is no decision to make
    -- and named, so it is still clear where the work will go.
    """
    try:
        workspaces = fetch(api_url, api_key)
    except Exception as e:
        console.print(f"[yellow]Could not list workspaces:[/yellow] {e}")
        console.print("  Choose one later with: corerun ws set")
        return default

    if not workspaces:
        console.print("[yellow]You are not a member of any workspace yet.[/yellow]")
        return None

    if len(workspaces) == 1:
        only = workspaces[0]
        console.print(f"  Workspace: [bold]{label(only)}[/bold]")
        return only["id"]

    _render(workspaces, current=default)

    # Default to the workspace already in use, so pressing Enter keeps things
    # as they are rather than moving the user somewhere.
    selected = 1
    for n, workspace in enumerate(workspaces, 1):
        if default and workspace["id"] == default:
            selected = n
            break

    from rich.prompt import IntPrompt

    choice = IntPrompt.ask(
        "[bold]Select a workspace[/bold]",
        choices=[str(n) for n in range(1, len(workspaces) + 1)],
        default=selected,
        show_choices=False,
    )
    chosen = workspaces[choice - 1]
    console.print(f"  Using [bold]{label(chosen)}[/bold]")
    return chosen["id"]


def _render(workspaces, current: str = None) -> None:
    table = Table(title="Workspaces", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Workspace")
    table.add_column("Tenant")
    table.add_column("Role")
    table.add_column("")

    for n, workspace in enumerate(workspaces, 1):
        table.add_row(
            str(n),
            workspace.get("display_name") or workspace.get("slug", ""),
            workspace.get("tenant_slug", ""),
            workspace.get("role", ""),
            "[green]current[/green]" if workspace.get("id") == current else "",
        )
    console.print()
    console.print(table)


@app.command("list")
def list_workspaces():
    """
    List the workspaces you belong to.

    Example:
        corerun ws list
    """
    config = _require_credentials()
    try:
        workspaces = fetch(config.api_url, config.auth_token)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    output.emit(
        workspaces,
        lambda: _render(workspaces, current=config.workspace)
        if workspaces
        else console.print("You are not a member of any workspace yet."),
    )


@app.command("set")
def set_workspace(
    name: str = typer.Argument(None, help="Workspace name, slug or ID; omitted, pick from a list"),
):
    """
    Choose the workspace commands act on.

    Example:
        corerun ws set ws1
        corerun ws set imys/ws1
        corerun ws set
    """
    config = _require_credentials()

    if name:
        try:
            workspaces = fetch(config.api_url, config.auth_token)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        chosen = resolve(workspaces, name)
        if chosen is None:
            console.print(f"[red]Error:[/red] no workspace called {name!r}")
            console.print("  Run 'corerun ws list' to see the ones you belong to.")
            raise typer.Exit(1)
        chosen_id = chosen["id"]
        console.print(f"  Using [bold]{label(chosen)}[/bold]")
    else:
        chosen_id = choose(config.api_url, config.auth_token, default=config.workspace)
        if not chosen_id:
            raise typer.Exit(1)

    config.workspace = chosen_id
    config.save()
    output.emit(
        {"workspace_id": chosen_id},
        lambda: console.print("[green]✓[/green] Workspace set"),
    )


@app.command("show")
def show_workspace():
    """
    Show which workspace commands currently act on.

    Example:
        corerun ws show
    """
    config = get_config()
    if not config.workspace:
        console.print("No workspace selected. Run 'corerun ws set'.")
        raise typer.Exit(1)

    try:
        workspaces = fetch(config.api_url, config.auth_token)
        current = next((w for w in workspaces if w["id"] == config.workspace), None)
    except Exception:
        current = None

    payload = current or {"id": config.workspace}
    output.emit(
        payload,
        lambda: console.print(
            f"[bold]{label(current)}[/bold]  ({current['id']}, role {current.get('role', '')})"
            if current
            else config.workspace
        ),
    )
