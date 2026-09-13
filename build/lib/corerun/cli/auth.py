"""
Authentication CLI commands
"""

import typer
from rich.console import Console

from corerun.cli import output
from corerun.cli import workspace as workspace_cli
from corerun.config import DEFAULT_API_URL, Config, init, get_config

console = output.console
app = typer.Typer(help="Authentication commands")


@app.command()
def login(
    auth_token: str = typer.Option(None, "--token", "-t", help="Auth token"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Default workspace ID"),
    api_url: str = typer.Option(None, "--url", help="API URL"),
    use_device_code: bool = typer.Option(
        False,
        "--use-device-code",
        help="Show a code to approve elsewhere instead of opening a browser",
    ),
):
    """
    Login to corerun and save credentials.

    Credentials are saved to ~/.corerun/config

    Example:
        corerun login --token <token> --workspace my-workspace-id
    """
    resolved_url = api_url or DEFAULT_API_URL

    # With no key given, sign in interactively rather than demanding one be
    # pasted.
    #
    # The browser is the default because it is the shorter path: the person is
    # usually already signed in there, and no code has to be copied between two
    # windows. The device flow is what a machine without a usable browser needs
    # -- an SSH session, a notebook terminal -- and is chosen automatically in
    # those cases, or explicitly with --use-device-code.
    refresh_token = None
    device_workspace = None
    if not auth_token:
        from corerun import browser_auth

        by_browser = not use_device_code and browser_auth.available()
        try:
            if by_browser:
                auth_token, refresh_token = _browser_login(resolved_url)
            else:
                auth_token, device_workspace, refresh_token = _device_login(resolved_url)
        except Exception as e:
            console.print(f"[red]Sign-in failed:[/red] {e}")
            if by_browser:
                console.print("Without a browser: corerun login --use-device-code")
            console.print("Or pass one directly: corerun login --token <token>")
            raise typer.Exit(1)

    # Which workspace to work in.
    #
    # Asked rather than assumed. Someone who belongs to several is otherwise
    # silently put in one of them, and every later command -- listing models,
    # deploying, pulling -- quietly acts on the wrong one. The device grant no
    # longer names one at all: the token is bound to the person, and the
    # workspace travels per request.
    if not workspace:
        workspace = workspace_cli.choose(resolved_url, auth_token, default=device_workspace)

    # Create config
    config = Config(
        auth_token=auth_token,
        refresh_token=refresh_token,
        workspace=workspace,
        api_url=resolved_url,
    )

    # Test connection
    try:
        console.print("Verifying credentials...", style="dim")
        client = init(
            auth_token=config.auth_token,
            workspace=config.workspace,
            api_url=config.api_url,
        )
        # Try to list datasets as a test
        client.get("/data")
        console.print("[green]✓[/green] Successfully authenticated!")
    except Exception as e:
        console.print(f"[red]✗[/red] Authentication failed: {e}")
        raise typer.Exit(1)

    # Save config
    config.save()
    console.print(f"[green]✓[/green] Credentials saved to ~/.corerun/config")

    if config.workspace:
        console.print(f"  Workspace: {config.workspace}")


@app.command()
def logout():
    """
    Remove saved credentials.
    """
    from pathlib import Path

    config_path = Path.home() / ".corerun" / "config"
    if config_path.exists():
        config_path.unlink()
        console.print("[green]✓[/green] Logged out successfully")
    else:
        console.print("No saved credentials found")


@app.command()
def whoami():
    """
    Show current authentication status.
    """
    config = get_config()

    if not config.auth_token:
        console.print("[yellow]Not logged in[/yellow]")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)

    # Mask auth token
    masked_key = config.auth_token[:10] + "..." + config.auth_token[-4:]

    console.print("[bold]corerun Authentication[/bold]")
    console.print(f"  Auth token: {masked_key}")
    console.print(f"  API URL: {config.api_url}")
    if config.workspace:
        console.print(f"  Workspace: {config.workspace}")

    # Test connection
    try:
        from corerun import init
        client = init()
        client.get("/data")
        console.print("  Status: [green]Connected[/green]")
    except Exception as e:
        console.print(f"  Status: [red]Error - {e}[/red]")


@app.command()
def status():
    """
    Show connection status (alias for whoami).
    """
    whoami()


def _web_url(api_url: str) -> str:
    """The site behind an API URL.

    The sign-in page lives with the rest of the product, not under /api/v1, so
    the CLI trims that off rather than being configured with the same host twice.
    """
    trimmed = api_url.rstrip("/")
    for suffix in ("/api/v1", "/api"):
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def _browser_login(api_url: str):
    """
    Sign in by opening a browser, and return the issued tokens.

    Nothing secret passes through the browser: the code it carries back is only
    usable together with a verifier this process keeps to itself.
    """
    from corerun import browser_auth

    def announce(url: str) -> None:
        console.print()
        console.print("  Opening your browser to sign in.")
        console.print("  If it does not open, use this link:")
        console.print(f"    [bold]{url}[/bold]")
        console.print()
        console.print("  Waiting...", style="dim")

    tokens = browser_auth.login(api_url, _web_url(api_url), on_url=announce)
    console.print("  [green]Signed in.[/green]")
    return tokens.get("access_token"), tokens.get("refresh_token")


def _device_login(api_url: str):
    """
    Sign in without a browser on this machine.

    Prints a short code, waits while the user approves it elsewhere, and returns
    the issued key and the workspace it was approved for.
    """
    import socket

    from corerun import device_auth

    grant = device_auth.start(api_url, client_name=socket.gethostname())

    console.print()
    # The address and the code, separately, rather than one link carrying both.
    #
    # RFC 8628 defines verification_uri_complete and the server sends it, but
    # leading with it here would be a mistake. The attack this flow is exposed
    # to is not someone guessing a code -- approving one grants their own access
    # to a terminal they do not control -- it is someone being sent a code and
    # approving it, which hands the sender a session as them. A one-click link
    # is that message, ready to forward. Typing the code is a small cost paid
    # for the moment of intent it forces, which is why Entra does not offer the
    # complete URI at all.
    console.print("  Open this URL and enter the code:")
    console.print(f"    [bold]{grant['verification_uri']}[/bold]")
    console.print(f"    Code: [bold cyan]{grant['user_code']}[/bold cyan]")
    console.print()
    console.print("  Waiting for approval...", style="dim")

    result = device_auth.poll(
        api_url,
        grant["device_code"],
        interval=grant.get("interval", 5),
    )
    console.print("  [green]Approved.[/green]")
    # access_token is the same value as auth_token; both are returned so a client
    # built before refresh tokens existed still finds what it expects.
    #
    # No workspace comes back any more. Approving a terminal says who it belongs
    # to, not where it will work, so the caller asks -- which it did regardless,
    # since the grant's workspace only named wherever the browser happened to be.
    return (
        result.get("access_token") or result["api_key"],
        result.get("workspace_id"),
        result.get("refresh_token"),
    )
