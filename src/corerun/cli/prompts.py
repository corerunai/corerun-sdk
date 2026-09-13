"""
Prompts CLI commands
"""

import typer
from rich.console import Console

from corerun.cli import output
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
import json

console = output.console
app = typer.Typer(help="Prompt registry commands")


def _init_client():
    """Initialize client, handling errors gracefully"""
    try:
        from corerun import init
        return init()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'corerun login' to authenticate")
        raise typer.Exit(1)


@app.command("list")
def list_prompts(
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search term"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Filter by tags (comma-separated)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List all prompts in the workspace.

    Example:
        corerun prompts list
        corerun prompts list --search summarization
        corerun prompts list --tags production,nlp
    """
    _init_client()

    import corerun.prompts as prompts

    try:
        tag_list = tags.split(",") if tags else None
        prompt_list = prompts.list(search=search, tags=tag_list, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        payload = []
        for p in prompt_list:
            payload.append({
                "prompt_id": p.prompt_id,
                "name": p.name,
                "description": p.description,
                "tags": p.tags,
                "latest_version": p.latest_version,
                "aliases": [{"alias": a.alias, "version": a.version} for a in p.aliases],
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            })
        output.emit(payload)
        return

    if not prompt_list:
        console.print("No prompts found")
        return

    table = Table(title="Prompts")
    table.add_column("Name", style="cyan")
    table.add_column("Version", justify="center")
    table.add_column("Aliases", style="green")
    table.add_column("Tags")
    table.add_column("Updated")

    for p in prompt_list:
        aliases_str = ", ".join(f"{a.alias}(v{a.version})" for a in p.aliases) if p.aliases else "-"
        tags_str = ", ".join(p.tags[:3]) if p.tags else "-"
        if p.tags and len(p.tags) > 3:
            tags_str += f" +{len(p.tags) - 3}"

        table.add_row(
            p.name,
            f"v{p.latest_version}",
            aliases_str,
            tags_str,
            p.updated_at.strftime("%Y-%m-%d") if p.updated_at else "-",
        )

    console.print(table)


@app.command("get")
def get_prompt(
    name: str = typer.Argument(..., help="Prompt name or ID"),
    version: Optional[int] = typer.Option(None, "--version", "-v", help="Specific version"),
    alias: Optional[str] = typer.Option(None, "--alias", "-a", help="Alias (e.g., production)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Get prompt details and template.

    Example:
        corerun prompts get summarization-prompt
        corerun prompts get summarization-prompt --alias production
        corerun prompts get summarization-prompt --version 2
    """
    _init_client()

    import corerun.prompts as prompts

    try:
        prompt = prompts.load(name, version=version, alias=alias, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        payload = {
            "prompt_id": prompt.prompt_id,
            "name": prompt.name,
            "description": prompt.description,
            "tags": prompt.tags,
            "latest_version": prompt.latest_version,
        }
        if prompt.version:
            payload["loaded_version"] = {
                "version": prompt.version.version,
                "prompt_type": prompt.version.prompt_type,
                "template": prompt.version.template,
                "messages": [{"role": m.role, "content": m.content} for m in prompt.version.messages],
                "variables": prompt.version.variables,
                "commit_message": prompt.version.commit_message,
            }
        output.emit(payload)
        return

    console.print(f"[bold]Prompt: {prompt.name}[/bold]")
    console.print(f"  ID: {prompt.prompt_id}")
    console.print(f"  Latest Version: v{prompt.latest_version}")

    if prompt.description:
        console.print(f"  Description: {prompt.description}")
    if prompt.tags:
        console.print(f"  Tags: {', '.join(prompt.tags)}")
    if prompt.aliases:
        aliases_str = ", ".join(f"{a.alias}(v{a.version})" for a in prompt.aliases)
        console.print(f"  Aliases: {aliases_str}")

    console.print()

    if prompt.version:
        v = prompt.version
        console.print(f"[bold]Loaded Version: v{v.version}[/bold]")
        console.print(f"  Type: {v.prompt_type}")
        if v.commit_message:
            console.print(f"  Commit: {v.commit_message}")
        if v.variables:
            console.print(f"  Variables: {', '.join(v.variables)}")
        console.print()

        if v.prompt_type == "text" and v.template:
            console.print(Panel(
                Syntax(v.template, "text", theme="monokai", word_wrap=True),
                title="Template",
            ))
        elif v.messages:
            for msg in v.messages:
                style = {
                    "system": "yellow",
                    "user": "cyan",
                    "assistant": "green",
                }.get(msg.role, "white")
                console.print(f"[{style}]{msg.role}:[/{style}]")
                console.print(Panel(msg.content, border_style=style))


@app.command("create")
def create_prompt(
    name: str = typer.Argument(..., help="Prompt name"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Template string (for text prompts)"),
    template_file: Optional[str] = typer.Option(None, "--file", "-f", help="Template file path"),
    prompt_type: str = typer.Option("chat", "--type", help="Prompt type: text or chat"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Tags (comma-separated)"),
    message: str = typer.Option("Initial version", "--message", "-m", help="Commit message"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Create a new prompt.

    Example:
        corerun prompts create my-prompt --template "Hello {{name}}"
        corerun prompts create my-prompt --file prompt.txt
        corerun prompts create my-prompt --type chat --file messages.json
    """
    _init_client()

    import corerun.prompts as prompts

    # Load template from file if provided
    content = template
    messages = None

    if template_file:
        try:
            with open(template_file, "r") as f:
                content = f.read()
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] File not found: {template_file}")
            raise typer.Exit(1)

        # Check if it's JSON (for chat messages)
        if template_file.endswith(".json"):
            try:
                messages = json.loads(content)
                prompt_type = "chat"
                content = None
            except json.JSONDecodeError:
                pass

    if not content and not messages:
        console.print("[red]Error:[/red] Provide --template, --file, or use interactive mode")
        raise typer.Exit(1)

    try:
        tag_list = tags.split(",") if tags else None
        prompt = prompts.create(
            name=name,
            prompt_type=prompt_type,
            template=content if prompt_type == "text" else None,
            messages=messages if prompt_type == "chat" else None,
            description=description or "",
            tags=tag_list,
            commit_message=message,
            workspace=workspace,
        )
        console.print(f"[green]Created prompt:[/green] {prompt.name} (ID: {prompt.prompt_id})")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("version")
def create_version(
    prompt_id: str = typer.Argument(..., help="Prompt ID or name"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="Template string"),
    template_file: Optional[str] = typer.Option(None, "--file", "-f", help="Template file path"),
    message: str = typer.Option(..., "--message", "-m", help="Commit message"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Create a new version of an existing prompt.

    Example:
        corerun prompts version my-prompt --template "Updated: {{text}}" --message "Improved wording"
        corerun prompts version my-prompt --file updated.json --message "Fixed system prompt"
    """
    _init_client()

    import corerun.prompts as prompts

    content = template
    messages = None

    if template_file:
        try:
            with open(template_file, "r") as f:
                content = f.read()
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] File not found: {template_file}")
            raise typer.Exit(1)

        if template_file.endswith(".json"):
            try:
                messages = json.loads(content)
                content = None
            except json.JSONDecodeError:
                pass

    try:
        version = prompts.create_version(
            prompt_id=prompt_id,
            template=content,
            messages=messages,
            commit_message=message,
            workspace=workspace,
        )
        console.print(f"[green]Created version:[/green] v{version.version}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("versions")
def list_versions(
    prompt_id: str = typer.Argument(..., help="Prompt ID or name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List all versions of a prompt.

    Example:
        corerun prompts versions summarization-prompt
    """
    _init_client()

    import corerun.prompts as prompts

    try:
        versions = prompts.list_versions(prompt_id, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        payload = []
        for v in versions:
            payload.append({
                "version": v.version,
                "prompt_type": v.prompt_type,
                "variables": v.variables,
                "commit_message": v.commit_message,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            })
        output.emit(payload)
        return

    if not versions:
        console.print("No versions found")
        return

    table = Table(title=f"Versions of {prompt_id}")
    table.add_column("Version", justify="center")
    table.add_column("Type")
    table.add_column("Variables")
    table.add_column("Commit Message")
    table.add_column("Created")

    for v in versions:
        vars_str = ", ".join(v.variables) if v.variables else "-"
        table.add_row(
            f"v{v.version}",
            v.prompt_type,
            vars_str,
            v.commit_message[:50] + "..." if len(v.commit_message) > 50 else v.commit_message,
            v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "-",
        )

    console.print(table)


@app.command("alias")
def set_alias(
    prompt_id: str = typer.Argument(..., help="Prompt ID or name"),
    alias: str = typer.Argument(..., help="Alias name (e.g., production, staging)"),
    version: int = typer.Argument(..., help="Version number"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Set an alias to point to a specific version.

    Example:
        corerun prompts alias summarization-prompt production 3
        corerun prompts alias my-prompt staging 2
    """
    _init_client()

    import corerun.prompts as prompts

    try:
        prompts.set_alias(prompt_id, alias, version, workspace=workspace)
        console.print(f"[green]Set alias:[/green] {alias} -> v{version}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("unalias")
def remove_alias(
    prompt_id: str = typer.Argument(..., help="Prompt ID or name"),
    alias: str = typer.Argument(..., help="Alias name to remove"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
):
    """
    Remove an alias from a prompt.

    Example:
        corerun prompts unalias summarization-prompt staging
    """
    _init_client()

    import corerun.prompts as prompts

    try:
        prompts.remove_alias(prompt_id, alias, workspace=workspace)
        console.print(f"[green]Removed alias:[/green] {alias}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("render")
def render_prompt(
    prompt_id: str = typer.Argument(..., help="Prompt ID or name"),
    variables: Optional[str] = typer.Option(None, "--vars", "-V", help="Variables as JSON object"),
    alias: Optional[str] = typer.Option(None, "--alias", "-a", help="Alias (e.g., production)"),
    version: Optional[int] = typer.Option(None, "--version", "-v", help="Specific version"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Render a prompt with variable substitution.

    Example:
        corerun prompts render summarization-prompt --vars '{"document": "Hello world"}'
        corerun prompts render my-prompt --alias production --vars '{"name": "John"}'
    """
    _init_client()

    import corerun.prompts as prompts

    try:
        prompt = prompts.load(prompt_id, version=version, alias=alias, workspace=workspace)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    vars_dict = {}
    if variables:
        try:
            vars_dict = json.loads(variables)
        except json.JSONDecodeError:
            console.print("[red]Error:[/red] Invalid JSON for --vars")
            raise typer.Exit(1)

    try:
        rendered = prompt.render(**vars_dict)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        output.set_json(True)
    if output.json_mode():
        if isinstance(rendered, str):
            output.emit({"rendered": rendered})
        else:
            output.emit({"messages": rendered})
        return

    if isinstance(rendered, str):
        console.print(Panel(rendered, title="Rendered Prompt"))
    else:
        for msg in rendered:
            style = {
                "system": "yellow",
                "user": "cyan",
                "assistant": "green",
            }.get(msg["role"], "white")
            console.print(f"[{style}]{msg['role']}:[/{style}]")
            console.print(Panel(msg["content"], border_style=style))


@app.command("delete")
def delete_prompt(
    prompt_id: str = typer.Argument(..., help="Prompt ID or name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Delete a prompt and all its versions.

    Example:
        corerun prompts delete my-prompt
        corerun prompts delete my-prompt --force
    """
    _init_client()

    import corerun.prompts as prompts

    if not force:
        confirm = typer.confirm(f"Delete prompt '{prompt_id}' and all versions?")
        if not confirm:
            console.print("Cancelled")
            raise typer.Exit(0)

    try:
        prompts.delete(prompt_id, workspace=workspace)
        console.print(f"[green]Deleted prompt:[/green] {prompt_id}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
