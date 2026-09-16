"""The skills may only teach commands the CLI actually has.

A skill is read as fact. An agent that tries `corerun models stage` and is told
"No such command" has no way to know which half of the skill to trust, so it
either stops or invents something -- and that command was documented here for a
while before anything implemented it.

This walks every `corerun <group> <sub>` in the skills and checks it against the
Typer app itself, rather than against a list someone has to remember to update.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer

from corerun.cli import app

SKILLS = Path(__file__).resolve().parents[1] / "src" / "corerun" / "skills"

def _names(entry) -> str | None:
    """What a registered command or group is called on the command line."""
    if entry.name:
        return entry.name
    instance = getattr(entry, "typer_instance", None)
    if instance is not None and instance.info.name:
        return instance.info.name
    callback = getattr(entry, "callback", None)
    return callback.__name__ if callback else None


def _groups() -> dict[str, set[str]]:
    """Every command group in the CLI, and what can follow it.

    Both commands and nested groups: `corerun datasets import` is a group of its
    own, not a command, and a check that knew only about commands called it
    missing.
    """
    found: dict[str, set[str]] = {}
    for group in app.registered_groups:
        name = _names(group)
        instance = getattr(group, "typer_instance", None)
        if not name or instance is None:
            continue
        children = {_names(c) for c in instance.registered_commands}
        children |= {_names(g) for g in instance.registered_groups}
        found[name] = {c for c in children if c}
    for command in app.registered_commands:
        name = _names(command)
        if name:
            found.setdefault(name, set())
    return found


# Only fenced blocks. Prose says things like "corerun runs the job" and
# "# corerun inference servers", and treating those as commands meant the check
# either cried wolf or needed a list of English words to forgive -- which would
# have forgiven a real command group that happened to be on it.
FENCE = re.compile(r"```[a-z]*\n(.*?)```", re.S)


def _references() -> list[tuple[Path, str, str]]:
    """(file, group, subcommand) for every command written in a skill."""
    out = []
    for path in sorted(SKILLS.rglob("*.md")):
        for block in FENCE.findall(path.read_text(encoding="utf-8")):
            for match in re.finditer(r"^\s*corerun\s+([a-z][a-z-]*)(?:\s+([a-z][a-z-]*))?", block, re.M):
                out.append((path, match.group(1), match.group(2) or ""))
    return out


def test_skills_reference_real_command_groups():
    groups = _groups()
    bad = {
        f"{path.parent.name}: corerun {group}"
        for path, group, _ in _references()
        if group not in groups
    }
    assert not bad, "skills name command groups the CLI does not have: " + ", ".join(sorted(bad))


def test_skills_reference_real_subcommands():
    groups = _groups()
    bad = set()
    for path, group, sub in _references():
        if not sub or group not in groups:
            continue
        known = groups[group]
        # A group with no registered subcommands takes arguments instead, so
        # the next word is a value and there is nothing to check.
        if known and sub not in known:
            bad.add(f"{path.parent.name}: corerun {group} {sub}")
    assert not bad, "skills name subcommands the CLI does not have: " + ", ".join(sorted(bad))


def test_every_skill_has_a_description():
    for path in sorted(SKILLS.rglob("SKILL.md")):
        head = path.read_text(encoding="utf-8").split("---")[1]
        assert "description:" in head, f"{path.parent.name} has no front-matter description"
        # The description is the whole interface: it is all an agent sees when
        # deciding whether to read the skill.
        description = re.search(r"description:\s*(.+)", head)
        assert description and len(description.group(1)) > 60, (
            f"{path.parent.name}'s description is too short to match against"
        )


def test_readme_lists_every_skill():
    readme = (SKILLS / "README.md").read_text(encoding="utf-8")
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        assert f"`{path.parent.name}`" in readme, f"{path.parent.name} is not in the README table"


@pytest.mark.parametrize("stage", ["none", "staging", "production", "archived"])
def test_documented_stages_are_the_api_s(stage):
    """The registry accepts these four and no others -- notably no 'development',
    which the models skill claimed for a while."""
    from corerun.cli.registry import STAGES

    assert stage in STAGES
    assert "development" not in STAGES
    assert typer  # the CLI imports cleanly, which is what makes the rest meaningful
