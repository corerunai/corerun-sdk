"""
GenAI observability CLI commands: agent traces, sessions and retention.
"""

from typing import Optional

import typer
from rich.table import Table

from corerun.cli import output
from corerun.exceptions import CoreRunError

console = output.console
app = typer.Typer(help="Agent traces, sessions and retention")

traces_app = typer.Typer(help="Agent traces")
sessions_app = typer.Typer(help="Conversations, grouped by session id")
app.add_typer(traces_app, name="traces")
app.add_typer(sessions_app, name="sessions")


def _init_client():
    try:
        from corerun import init

        return init()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)


def _called(work):
    """Run a call to the service, reporting a failure the way the CLI does.

    A missing trace is an ordinary answer to an ordinary question, not a crash,
    and a stack trace in front of it helps nobody holding a trace id that has
    aged out of retention.
    """
    try:
        return work()
    except CoreRunError as e:
        raise output.fail(str(e))


def _duration(ms: Optional[float]) -> str:
    """The unit follows the magnitude, so 20.31s and 60ms sit in one column."""
    if ms is None:
        return "-"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f}s"
    return f"{ms / 60_000:.1f}min"


def _short(text: Optional[str], width: int = 40) -> str:
    if not text:
        return "-"
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


@traces_app.command("list")
def list_traces(
    state: Optional[str] = typer.Option(None, "--state", help="OK, ERROR or IN_PROGRESS"),
    session: Optional[str] = typer.Option(None, "--session", help="Only one conversation"),
    model: Optional[str] = typer.Option(None, "--model", help="Only traces that asked this model"),
    search: Optional[str] = typer.Option(None, "--search", help="Match id, input or output"),
    since: Optional[str] = typer.Option(None, "--since", help="RFC 3339 timestamp"),
    limit: int = typer.Option(25, "--limit"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """List agent traces, most recent first."""
    _init_client()
    from corerun import genai

    found = _called(lambda: genai.traces(
        state=state,
        session_id=session,
        model=model,
        search=search,
        since=since,
        limit=limit,
        workspace=workspace,
    ))

    def render():
        if not found:
            console.print("[dim]No traces. Point an instrumented application at this workspace.[/dim]")
            return
        table = Table(show_header=True, header_style="bold")
        table.add_column("Trace")
        table.add_column("Root span")
        table.add_column("State")
        table.add_column("Duration", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Session")
        for t in found:
            table.add_row(
                t.trace_id[:16],
                _short(t.root_span_name, 28),
                "[red]ERROR[/red]" if t.failed else (t.state or "-"),
                _duration(t.duration_ms),
                f"{t.total_tokens:,}" if t.total_tokens else "-",
                t.session_id or "-",
            )
        console.print(table)

    output.emit([t.__dict__ for t in found], render)


@traces_app.command("get")
def get_trace(
    trace_id: str = typer.Argument(..., help="The trace id"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """Show a trace and its span tree."""
    _init_client()
    from corerun import genai

    detail = _called(lambda: genai.trace(trace_id, workspace=workspace))

    def render():
        console.print(f"[bold]{detail.root_span_name or detail.trace_id}[/bold]  {detail.trace_id}")
        console.print(
            f"  {detail.state or '-'}   {_duration(detail.duration_ms)}"
            f"   {f'{detail.total_tokens:,} tokens' if detail.total_tokens else ''}"
        )
        if detail.session_id:
            console.print(f"  session {detail.session_id}")
        console.print()

        # Indented by depth, so the tree reads as a tree in a terminal too.
        children: dict = {}
        for span in detail.spans:
            children.setdefault(span.parent_span_id, []).append(span)
        known = {s.span_id for s in detail.spans}

        def walk(parent, depth):
            for span in children.get(parent, []):
                console.print(
                    f"{'  ' * depth}{span.name}  [dim]{_duration(span.duration_ms)}"
                    f"{'  ' + format(span.total_tokens, ',') + ' tok' if span.total_tokens else ''}[/dim]"
                )
                walk(span.span_id, depth + 1)

        # A span whose parent is missing is a root here too: exports are
        # batched, so a child can outrun its parent.
        for parent in [None] + [p for p in children if p is not None and p not in known]:
            walk(parent, 0)

    payload = detail.__dict__.copy()
    payload["spans"] = [s.__dict__ for s in detail.spans]
    output.emit(payload, render)


@traces_app.command("delete")
def delete_trace(
    trace_id: str = typer.Argument(..., help="The trace id"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """Delete a trace and everything recorded about it."""
    _init_client()
    from corerun import genai

    if not yes and not typer.confirm(f"Delete trace {trace_id} and everything recorded about it?"):
        raise typer.Exit(0)
    _called(lambda: genai.delete_trace(trace_id, workspace=workspace))
    output.emit({"deleted": trace_id}, lambda: console.print(f"[green]Deleted[/green] {trace_id}"))


@sessions_app.command("list")
def list_sessions(
    limit: int = typer.Option(25, "--limit"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """List conversations, most recently active first."""
    _init_client()
    from corerun import genai

    found = _called(lambda: genai.sessions(limit=limit, workspace=workspace))

    def render():
        if not found:
            console.print("[dim]No sessions. A session appears once traces carry a session id.[/dim]")
            return
        table = Table(show_header=True, header_style="bold")
        table.add_column("Session")
        table.add_column("Traces", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Last active")
        for s in found:
            table.add_row(
                s.session_id,
                str(s.trace_count),
                f"[red]{s.error_count}[/red]" if s.error_count else "-",
                f"{s.total_tokens:,}" if s.total_tokens else "-",
                s.last_seen or "-",
            )
        console.print(table)

    output.emit([s.__dict__ for s in found], render)


@app.command("settings")
def show_settings(
    retention_days: Optional[int] = typer.Option(None, "--retention-days", help="Set retention"),
    sample_rate: Optional[float] = typer.Option(None, "--sample-rate", help="Set sampling, 0 to 1"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """Show or change what this workspace keeps."""
    _init_client()
    from corerun import genai

    if retention_days is None and sample_rate is None:
        current = _called(lambda: genai.settings(workspace=workspace))
    else:
        current = _called(
            lambda: genai.configure(
                retention_days=retention_days, sample_rate=sample_rate, workspace=workspace
            )
        )

    def render():
        kept = "kept until deleted" if not current.retention_days else f"{current.retention_days} days"
        console.print(f"Retention:  {kept}")
        console.print(f"Sampling:   {current.sample_rate:g}")
        if current.updated_by:
            console.print(f"[dim]last changed by {current.updated_by} {current.updated_at or ''}[/dim]")

    output.emit(current.__dict__, render)
