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
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """List agent traces, most recent first."""
    if json_output:
        output.set_json(True)
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
        # Shortened, but only because a short id is now a valid one: `get` and
        # `delete` resolve a prefix. A whole 32-character id in an 80-column
        # terminal squeezes every other column to nothing, and the id was the
        # only part of the row that survived -- which is the wrong trade when
        # the rest of the row is what tells you which trace you want.
        table.add_column("Trace", no_wrap=True)
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


# The eighths, so a bar can end part way through a cell. A span that ran for a
# fifth of a column still gets drawn rather than rounded away to nothing.
_EIGHTHS = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589\u2588"


def _bar(start: float, end: float, width: int) -> str:
    """One span drawn on the trace's clock, as a bar in `width` columns.

    Position carries as much as length does: two slow steps that overlapped are
    a different problem from two that ran one after the other, and a list of
    durations cannot tell you which you have.
    """
    if width < 4:
        return ""
    start = max(0.0, min(1.0, start))
    end = max(start, min(1.0, end))

    left = start * width
    right = end * width
    cells = []
    for i in range(width):
        covered = min(right, i + 1) - max(left, i)
        if covered <= 0:
            cells.append(" ")
        else:
            cells.append(_EIGHTHS[max(1, min(8, round(covered * 8)))])
    drawn = "".join(cells)
    # A span too short to fill an eighth still happened, so it gets a mark.
    return drawn if drawn.strip() else " " * min(int(left), width - 1) + "\u258f"


# Past this the columns drift apart rather than becoming easier to read.
LAYOUT_MAX_WIDTH = 104


def _span_lines(detail, width: int):
    """The span tree as lines: guides, name, timings, and a bar.

    Drawn by hand rather than with rich's Tree because the columns have to line
    up across depths, and a tree puts its guides before the label -- so a label
    padded to a fixed width still starts in a different column on every level.
    """
    children: dict = {}
    for span in detail.spans:
        children.setdefault(span.parent_span_id, []).append(span)
    known = {s.span_id for s in detail.spans}
    # A span whose parent is missing is a root: exports are batched, so a child
    # can outrun its parent, and dropping it would hide a step that happened.
    roots = [None] + [p for p in children if p is not None and p not in known]

    ordered = []

    def walk(parent, prefix, depth):
        kids = children.get(parent, [])
        for i, span in enumerate(kids):
            last = i == len(kids) - 1
            joint = "\u2514\u2500 " if last else "\u251c\u2500 "
            ordered.append((prefix + joint if depth else "", span))
            # A root draws no guide, so its children start at the left edge
            # rather than inheriting three columns of padding from a joint
            # that was never drawn.
            below = "" if depth == 0 else prefix + ("   " if last else "\u2502  ")
            walk(span.span_id, below, depth + 1)

    for root in roots:
        walk(root, "", 0)

    if not ordered:
        return None

    def moment(iso):
        from datetime import datetime

        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() if iso else None

    starts = [moment(s.start_time) for _, s in ordered]
    starts = [m for m in starts if m is not None]
    origin = min(starts) if starts else 0.0
    total = max(((detail.duration_ms or 0) / 1000.0), 0.001)

    # Widths: the bar takes what it can, and the name column is only as wide as
    # the longest name in this trace. Padding every name out to the space
    # available leaves a gulf between a short name and its timings, which is
    # the same problem as letting the whole layout fill a wide terminal.
    bar_width = max(10, min(28, width - 52))
    longest = max(len(g + s.name) for g, s in ordered)
    name_width = max(16, min(longest, width - bar_width - 26))

    lines = []
    for guides, span in ordered:
        began = moment(span.start_time)
        offset = ((began - origin) / total) if began is not None else 0.0
        ran = (span.duration_ms or 0) / 1000.0 / total

        label = guides + span.name
        if len(label) > name_width:
            label = label[: name_width - 1] + "\u2026"

        failed = span.status == "ERROR"
        style = "bold" if failed else ""
        close = "[/bold]" if failed else ""
        mark = "!" if failed else " "
        tokens = f"{span.total_tokens:,}" if span.total_tokens else ""
        lines.append(
            f"{'[bold]' if failed else ''}{mark}{label:<{name_width}}{close}"
            f" [dim]{_duration(span.duration_ms):>9}[/dim]"
            f" [dim]{tokens:>8}[/dim]"
            f" {'[bold]' if failed else ''}{_bar(offset, offset + ran, bar_width)}{close}"
        )
    return lines, name_width, bar_width


def _resolve(trace_id: str, workspace: Optional[str]) -> str:
    """Accept a shortened id and return the whole one.

    A trace id is 32 hex characters, and every place that displays one shortens
    it -- the console shows eight in its chip. Somebody reading an id off a
    screen is holding a prefix, and answering "no trace with that id" to a
    prefix that names exactly one trace is a refusal on a technicality.

    Only a short id costs the extra lookup, and an ambiguous one says so rather
    than picking.
    """
    from corerun import genai

    if len(trace_id) >= 32:
        return trace_id

    matching = [
        t.trace_id
        for t in _called(lambda: genai.traces(search=trace_id, limit=50, workspace=workspace))
        if t.trace_id.startswith(trace_id)
    ]
    if not matching:
        raise output.fail(f"no trace starting with {trace_id} in this workspace")
    if len(matching) > 1:
        raise output.fail(
            f"{trace_id} names {len(matching)} traces: " + ", ".join(m[:16] for m in matching[:4])
        )
    return matching[0]


@traces_app.command("get")
def get_trace(
    trace_id: str = typer.Argument(..., help="The trace id, or enough of its start to be unique"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """Show a trace and its span tree."""
    if json_output:
        output.set_json(True)
    _init_client()
    from corerun import genai

    detail = _called(lambda: genai.trace(_resolve(trace_id, workspace), workspace=workspace))

    def render():
        # Capped rather than the full terminal: on a wide screen the bars end
        # up in the far corner, a gulf away from the names they belong to --
        # and a name beside its bar is the only reason they share a line.
        width = min(console.width, LAYOUT_MAX_WIDTH)

        state = (detail.state or "").upper()
        mark = "\u2717" if state == "ERROR" else "\u25cf" if state == "IN_PROGRESS" else "\u2713"
        label = {"ERROR": "Error", "IN_PROGRESS": "In progress"}.get(state, "Success")

        console.print()
        console.print(
            f"[bold]{detail.root_span_name or 'trace'}[/bold]  [dim]{detail.trace_id}[/dim]"
        )

        facts = [f"{mark} {label}", f"[dim]{_duration(detail.duration_ms)}[/dim]"]
        if detail.total_tokens:
            facts.append(f"[dim]{detail.total_tokens:,} tokens[/dim]")
        if detail.request_model:
            facts.append(f"[dim]{detail.request_model}[/dim]")
        if detail.session_id:
            facts.append(f"[dim]session {detail.session_id}[/dim]")
        console.print("  " + "  [dim]\u00b7[/dim]  ".join(facts))
        console.print()

        drawn = _span_lines(detail, width)
        if not drawn:
            console.print("[dim]This trace has no spans.[/dim]")
            return
        lines, name_width, bar_width = drawn
        console.print(
            f"[dim] {'SPAN':<{name_width}} {'DURATION':>9} {'TOKENS':>8} "
            f"{'0s':<{max(1, bar_width - len(_duration(detail.duration_ms)))}}"
            f"{_duration(detail.duration_ms)}[/dim]"
        )
        for line in lines:
            console.print(line)
        console.print()

    payload = detail.__dict__.copy()
    payload["spans"] = [s.__dict__ for s in detail.spans]
    output.emit(payload, render)


@traces_app.command("delete")
def delete_trace(
    trace_id: str = typer.Argument(..., help="The trace id"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """Delete a trace and everything recorded about it."""
    if json_output:
        output.set_json(True)
    _init_client()
    from corerun import genai

    trace_id = _resolve(trace_id, workspace)
    if not yes and not typer.confirm(f"Delete trace {trace_id} and everything recorded about it?"):
        raise typer.Exit(0)
    _called(lambda: genai.delete_trace(trace_id, workspace=workspace))
    output.emit({"deleted": trace_id}, lambda: console.print(f"[green]Deleted[/green] {trace_id}"))


@sessions_app.command("list")
def list_sessions(
    limit: int = typer.Option(25, "--limit"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """List conversations, most recently active first."""
    if json_output:
        output.set_json(True)
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
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
):
    """Show or change what this workspace keeps."""
    if json_output:
        output.set_json(True)
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
