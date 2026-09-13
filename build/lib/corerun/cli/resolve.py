"""Turning what a person typed into the id an API wants.

Every list command prints a name and a shortened id, and every other command
used to accept neither — only the full id, which appears nowhere on screen. So
the value shown in a table could not be pasted into the command beside it, and
the name everyone actually uses did not work at all.

Names are unique within a workspace for notebooks, jobs and inference servers
alike, which is what makes accepting one safe.
"""

import re
from typing import Any, Callable, List, Optional

import typer

from corerun.cli import output

console = output.console

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def by_name_or_id(reference: str, lister: Callable[[], List[Any]], label: str) -> str:
    """Resolve a name, a full id, or an unambiguous id prefix to an id.

    Args:
        reference: what the person typed
        lister: returns the candidates; each needs .id and .name
        label: what to call the thing in a message, e.g. "notebook"

    An ambiguous prefix is refused with the candidates rather than guessed at,
    and an unknown reference says what is actually there — "not_found" alone
    leaves someone to guess whether they mistyped or are in the wrong
    workspace.
    """
    if _UUID.match(reference):
        return reference

    try:
        found = lister()
    except Exception:
        # The lookup is a convenience. If it fails, let the real call report
        # why rather than replacing its error with one about listing.
        return reference

    if exact_id := next((i for i in found if getattr(i, "id", None) == reference), None):
        return exact_id.id

    # Names are unique to their owner, not to the workspace, so two visible
    # things can share one — yours and one somebody shared with you. Refusing
    # is better than picking: acting on the wrong notebook is worse than being
    # asked which.
    named = [i for i in found if getattr(i, "name", None) == reference]
    if len(named) == 1:
        return named[0].id
    if len(named) > 1:
        ids = ", ".join(i.id for i in named)
        console.print(
            f"[red]Error:[/red] more than one {label} is called '{reference}'. Use an id: {ids}"
        )
        raise typer.Exit(1)

    prefixed = [i for i in found if str(getattr(i, "id", "")).startswith(reference)]
    if len(prefixed) == 1:
        return prefixed[0].id
    if len(prefixed) > 1:
        names = ", ".join(f"{i.name} ({i.id})" for i in prefixed)
        console.print(f"[red]Error:[/red] '{reference}' matches more than one {label}: {names}")
        raise typer.Exit(1)

    known = ", ".join(str(getattr(i, "name", i.id)) for i in found) or "none"
    console.print(f"[red]Error:[/red] no {label} called '{reference}'. In this workspace: {known}")
    raise typer.Exit(1)
