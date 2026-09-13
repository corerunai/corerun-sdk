"""Installing the corerun skills into an agent's skills directory.

The skills ship inside this package so that anywhere the CLI is installed has
them: a laptop, CI, a notebook, an agent that has never seen corerun. They used
to live in the notebook image, which meant the only way to get them was to be
inside one.
"""

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from corerun.cli import output

console = output.console
app = typer.Typer(help="corerun skills for coding agents")

# Where each agent looks. The first that already exists is the default target,
# because a directory an agent created is better evidence of what it reads than
# anything this tool could guess.
KNOWN_SKILL_DIRS = [
    Path.home() / ".agents" / "skills",   # opencode, DSH, goose
    Path.home() / ".claude" / "skills",   # Claude Code
]


def _bundled() -> Path:
    """The skills carried in this package."""
    return Path(__file__).resolve().parent.parent / "skills"


def _available() -> list[Path]:
    root = _bundled()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "SKILL.md").is_file())


def _describe(skill: Path) -> str:
    """The description an agent matches against, read from the front matter."""
    try:
        lines = (skill / "SKILL.md").read_text().splitlines()
    except OSError:
        return ""
    for line in lines[:10]:
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


def _default_target() -> Path:
    for candidate in KNOWN_SKILL_DIRS:
        if candidate.is_dir():
            return candidate
    return KNOWN_SKILL_DIRS[0]


@app.command("list")
def list_skills():
    """Show the skills this CLI carries."""
    skills = _available()
    if not skills:
        console.print("[yellow]No skills are bundled with this build.[/yellow]")
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Skill")
    table.add_column("What it covers")
    for skill in skills:
        table.add_row(skill.name, _describe(skill))
    console.print(table)


@app.command("install")
def install(
    dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Where to install. Defaults to the first agent skills directory that exists.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Replace skills that are already there.",
    ),
):
    """Copy the corerun skills into an agent's skills directory."""
    skills = _available()
    if not skills:
        console.print("[red]Error:[/red] no skills are bundled with this build.")
        raise typer.Exit(1)

    target = (dir or _default_target()).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(f"[red]Error:[/red] could not create {target}: {e}")
        raise typer.Exit(1)

    installed, skipped = [], []
    for skill in skills:
        destination = target / skill.name
        if destination.exists() and not force:
            skipped.append(skill.name)
            continue
        try:
            # Replaced rather than merged: a half-updated skill is worse than
            # either version of it.
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(skill, destination)
        except OSError as e:
            console.print(f"[red]Error:[/red] could not install {skill.name}: {e}")
            raise typer.Exit(1)
        installed.append(skill.name)

    for name in installed:
        console.print(f"[green]installed[/green] {name}")
    for name in skipped:
        console.print(f"[yellow]already there[/yellow] {name}")

    console.print(f"\n{len(installed)} installed in {target}")
    if skipped:
        console.print("Run with --force to replace the ones already there.")
