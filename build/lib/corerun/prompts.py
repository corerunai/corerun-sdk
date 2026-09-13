"""
corerun Prompts Module

Provides access to the Prompt Registry for version-controlled prompt templates.

Usage:
    import corerun
    from corerun import prompts

    corerun.init()

    # Load a prompt by alias
    prompt = prompts.load("summarization-prompt", alias="production")
    messages = prompt.render(document="...", focus_areas="key points")

    # List prompts
    all_prompts = prompts.list()

    # Create a new prompt
    prompt = prompts.create(
        name="my-prompt",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Summarize: {{document}}"},
        ],
        commit_message="Initial version",
    )

    # Create a new version
    prompts.create_version(
        prompt_id="...",
        messages=[...],
        commit_message="Updated system prompt",
    )

    # Manage aliases
    prompts.set_alias("prompt-id", "production", version=2)
    prompts.remove_alias("prompt-id", "staging")
"""

import re
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """A chat message in a prompt template."""
    role: str
    content: str

    def render(self, variables: Dict[str, str]) -> Dict[str, str]:
        """Render the message with variable substitution."""
        content = self.content
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", value)
        return {"role": self.role, "content": content}


@dataclass
class PromptVersion:
    """A specific version of a prompt."""
    version_id: str
    prompt_id: str
    version: int
    prompt_type: str  # "text" or "chat"
    template: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    model_config: Dict[str, Any] = field(default_factory=dict)
    commit_message: str = ""
    created_by: str = ""
    created_at: Optional[datetime] = None

    def render(self, **variables: str) -> Union[str, List[Dict[str, str]]]:
        """Render the prompt with variable substitution.

        Args:
            **variables: Variable values to substitute

        Returns:
            For text prompts: rendered string
            For chat prompts: list of rendered messages
        """
        if self.prompt_type == "text" and self.template:
            result = self.template
            for key, value in variables.items():
                result = result.replace(f"{{{{{key}}}}}", value)
            return result
        else:
            return [msg.render(variables) for msg in self.messages]

    def render_messages(self, **variables: str) -> List[Dict[str, str]]:
        """Render as messages (for chat prompts)."""
        if self.prompt_type == "text" and self.template:
            return [{"role": "user", "content": self.render(**variables)}]
        return self.render(**variables)


@dataclass
class PromptAlias:
    """An alias pointing to a specific version."""
    alias: str
    version: int
    updated_at: Optional[datetime] = None


@dataclass
class Prompt:
    """A prompt with version history and aliases."""
    prompt_id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    latest_version: int = 1
    aliases: List[PromptAlias] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Current loaded version (set when loading)
    _current_version: Optional[PromptVersion] = field(default=None, repr=False)

    def render(self, **variables: str) -> Union[str, List[Dict[str, str]]]:
        """Render the current loaded version with variables."""
        if not self._current_version:
            raise ValueError("No version loaded. Use prompts.load() with version or alias.")
        return self._current_version.render(**variables)

    def render_messages(self, **variables: str) -> List[Dict[str, str]]:
        """Render as messages."""
        if not self._current_version:
            raise ValueError("No version loaded. Use prompts.load() with version or alias.")
        return self._current_version.render_messages(**variables)

    @property
    def version(self) -> Optional[PromptVersion]:
        """Get the currently loaded version."""
        return self._current_version


def _get_client():
    """Get the corerun client."""
    from corerun import get_client
    return get_client()


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _parse_prompt(data: Dict[str, Any]) -> Prompt:
    """Parse prompt data from API response."""
    aliases = []
    for alias_data in data.get("aliases", []):
        aliases.append(PromptAlias(
            alias=alias_data.get("alias", ""),
            version=alias_data.get("version", 0),
            updated_at=_parse_datetime(alias_data.get("updated_at")),
        ))

    return Prompt(
        prompt_id=data.get("prompt_id", ""),
        name=data.get("name", ""),
        description=data.get("description", ""),
        tags=data.get("tags", []),
        latest_version=data.get("latest_version", 1),
        aliases=aliases,
        created_at=_parse_datetime(data.get("created_at")),
        updated_at=_parse_datetime(data.get("updated_at")),
    )


def _parse_version(data: Dict[str, Any]) -> PromptVersion:
    """Parse prompt version data from API response."""
    messages = []
    for msg_data in data.get("messages", []):
        messages.append(Message(
            role=msg_data.get("role", "user"),
            content=msg_data.get("content", ""),
        ))

    return PromptVersion(
        version_id=data.get("version_id", ""),
        prompt_id=data.get("prompt_id", ""),
        version=data.get("version", 1),
        prompt_type=data.get("prompt_type", "chat"),
        template=data.get("template"),
        messages=messages,
        variables=data.get("variables", []),
        model_config=data.get("model_config", {}),
        commit_message=data.get("commit_message", ""),
        created_by=data.get("created_by", ""),
        created_at=_parse_datetime(data.get("created_at")),
    )


def list(
    search: Optional[str] = None,
    tags: Optional[List[str]] = None,
    workspace: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Prompt]:
    """List prompts in the workspace.

    Args:
        search: Search term for name/description
        tags: Filter by tags
        workspace: Workspace ID (uses default if not specified)
        limit: Maximum number of prompts to return
        offset: Pagination offset

    Returns:
        List of Prompt objects
    """
    client = _get_client()

    params = {"limit": str(limit), "offset": str(offset)}
    if search:
        params["search"] = search
    if tags:
        params["tags"] = ",".join(tags)

    response = client.get("/prompts", params=params, workspace=workspace)
    data = response

    return [_parse_prompt(p) for p in data.get("prompts", [])]


def get(
    prompt_id: str,
    workspace: Optional[str] = None,
) -> Prompt:
    """Get prompt details.

    Args:
        prompt_id: Prompt ID or name
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Prompt object
    """
    client = _get_client()
    response = client.get(f"/prompts/{prompt_id}", workspace=workspace)
    return _parse_prompt(response)


def load(
    prompt_id: str,
    version: Optional[int] = None,
    alias: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Prompt:
    """Load a prompt with a specific version or alias.

    Args:
        prompt_id: Prompt ID or name
        version: Specific version number
        alias: Alias name (e.g., "production", "staging")
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Prompt object with loaded version

    Example:
        # Load by alias
        prompt = prompts.load("summarization-prompt", alias="production")
        messages = prompt.render(document="Hello world")

        # Load specific version
        prompt = prompts.load("summarization-prompt", version=2)
    """
    client = _get_client()

    params = {}
    if version:
        params["version"] = str(version)
    elif alias:
        params["alias"] = alias

    response = client.get(f"/prompts/{prompt_id}/load", params=params, workspace=workspace)
    data = response

    # Parse the prompt and version together
    prompt = _parse_prompt(data)
    prompt._current_version = _parse_version(data)

    return prompt


def create(
    name: str,
    prompt_type: str = "chat",
    template: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    commit_message: str = "Initial version",
    workspace: Optional[str] = None,
) -> Prompt:
    """Create a new prompt.

    Args:
        name: Prompt name (must be unique in workspace)
        prompt_type: "text" or "chat"
        template: Template string for text prompts
        messages: Message array for chat prompts
        description: Prompt description
        tags: List of tags
        model_config: Optional model configuration (temperature, etc.)
        commit_message: Commit message for initial version
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Created Prompt object

    Example:
        # Create chat prompt
        prompt = prompts.create(
            name="summarization-prompt",
            messages=[
                {"role": "system", "content": "You summarize text concisely."},
                {"role": "user", "content": "Summarize: {{document}}"},
            ],
            tags=["summarization", "production"],
        )

        # Create text prompt
        prompt = prompts.create(
            name="completion-prompt",
            prompt_type="text",
            template="Complete the following: {{text}}",
        )
    """
    client = _get_client()

    payload = {
        "name": name,
        "prompt_type": prompt_type,
        "description": description,
        "tags": tags or [],
        "commit_message": commit_message,
    }

    if prompt_type == "text" and template:
        payload["template"] = template
    elif prompt_type == "chat" and messages:
        payload["messages"] = messages

    if model_config:
        payload["model_config"] = model_config

    response = client.post("/prompts", json=payload, workspace=workspace)
    return _parse_prompt(response)


def create_version(
    prompt_id: str,
    template: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    commit_message: str = "",
    workspace: Optional[str] = None,
) -> PromptVersion:
    """Create a new version of an existing prompt.

    Args:
        prompt_id: Prompt ID
        template: Template string for text prompts
        messages: Message array for chat prompts
        model_config: Optional model configuration
        commit_message: Commit message for this version
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Created PromptVersion object

    Example:
        version = prompts.create_version(
            prompt_id="...",
            messages=[
                {"role": "system", "content": "Improved system prompt..."},
                {"role": "user", "content": "{{question}}"},
            ],
            commit_message="Improved system prompt for better accuracy",
        )
    """
    client = _get_client()

    payload = {"commit_message": commit_message}

    if template:
        payload["template"] = template
    if messages:
        payload["messages"] = messages
    if model_config:
        payload["model_config"] = model_config

    response = client.post(f"/prompts/{prompt_id}/versions", json=payload, workspace=workspace)
    return _parse_version(response)


def list_versions(
    prompt_id: str,
    workspace: Optional[str] = None,
) -> List[PromptVersion]:
    """List all versions of a prompt.

    Args:
        prompt_id: Prompt ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        List of PromptVersion objects
    """
    client = _get_client()
    response = client.get(f"/prompts/{prompt_id}/versions", workspace=workspace)
    data = response

    return [_parse_version(v) for v in data.get("versions", [])]


def get_version(
    prompt_id: str,
    version: int,
    workspace: Optional[str] = None,
) -> PromptVersion:
    """Get a specific version of a prompt.

    Args:
        prompt_id: Prompt ID
        version: Version number
        workspace: Workspace ID (uses default if not specified)

    Returns:
        PromptVersion object
    """
    client = _get_client()
    response = client.get(f"/prompts/{prompt_id}/versions/{version}", workspace=workspace)
    return _parse_version(response)


def set_alias(
    prompt_id: str,
    alias: str,
    version: int,
    workspace: Optional[str] = None,
) -> None:
    """Set an alias to point to a specific version.

    Args:
        prompt_id: Prompt ID
        alias: Alias name (e.g., "production", "staging", "latest")
        version: Version number to point to
        workspace: Workspace ID (uses default if not specified)

    Example:
        # Promote version 3 to production
        prompts.set_alias("my-prompt", "production", version=3)
    """
    client = _get_client()
    client.put(
        f"/prompts/{prompt_id}/aliases/{alias}",
        json={"version": version},
        workspace=workspace,
    )


def remove_alias(
    prompt_id: str,
    alias: str,
    workspace: Optional[str] = None,
) -> None:
    """Remove an alias from a prompt.

    Args:
        prompt_id: Prompt ID
        alias: Alias name to remove
        workspace: Workspace ID (uses default if not specified)
    """
    client = _get_client()
    client.delete(f"/prompts/{prompt_id}/aliases/{alias}", workspace=workspace)


def list_aliases(
    prompt_id: str,
    workspace: Optional[str] = None,
) -> List[PromptAlias]:
    """List all aliases for a prompt.

    Args:
        prompt_id: Prompt ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        List of PromptAlias objects
    """
    client = _get_client()
    response = client.get(f"/prompts/{prompt_id}/aliases", workspace=workspace)
    data = response

    aliases = []
    for alias_data in data.get("aliases", []):
        aliases.append(PromptAlias(
            alias=alias_data.get("alias", ""),
            version=alias_data.get("version", 0),
            updated_at=_parse_datetime(alias_data.get("updated_at")),
        ))

    return aliases


def delete(
    prompt_id: str,
    workspace: Optional[str] = None,
) -> None:
    """Delete a prompt and all its versions.

    Args:
        prompt_id: Prompt ID
        workspace: Workspace ID (uses default if not specified)
    """
    client = _get_client()
    client.delete(f"/prompts/{prompt_id}", workspace=workspace)


def extract_variables(text: str) -> List[str]:
    """Extract variable names from a template string.

    Variables are in the format {{variable_name}}.

    Args:
        text: Template string

    Returns:
        List of unique variable names

    Example:
        vars = extract_variables("Hello {{name}}, you have {{count}} messages")
        # Returns: ["name", "count"]
    """
    matches = re.findall(r"\{\{(\w+)\}\}", text)
    return list(dict.fromkeys(matches))  # Preserve order, remove duplicates
