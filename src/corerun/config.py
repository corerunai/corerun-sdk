"""
Configuration management for corerun SDK.

Supports configuration via:
1. Direct initialization: corerun.init(auth_token="...", workspace="...")
2. Environment variables: CORERUN_AUTH_TOKEN, CORERUN_WORKSPACE, CORERUN_API_URL
3. Config file: ~/.corerun/config
4. Platform credential: /etc/corerun/api-key, installed in corerun notebooks

The platform credential is last so that anything the user chose themselves —
`corerun login`, an explicit key — wins over the one the notebook was handed.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from corerun.credentials import credential_path, read_platform_credential



# Where a client looks unless it is told otherwise.
#
# The platform's own address and, separately, the host model endpoints are
# published on. Both are overridden by ~/.corerun/config or the matching
# environment variable, which is what a self-hosted deployment does -- these
# are what corerun.ai serves, not an assumption that everybody uses it.
DEFAULT_API_URL = "https://corerun.ai/api/v1"
DEFAULT_INFERENCE_URL = "https://api.corerun.ai"


@dataclass
class Config:
    """corerun configuration"""
    auth_token: Optional[str] = None

    # Exchanged for a new auth_token when the current one expires. Access tokens
    # are short-lived on purpose; without this, signing in would be a daily
    # chore rather than something done once.
    refresh_token: Optional[str] = None

    workspace: Optional[str] = None
    api_url: str = DEFAULT_API_URL

    # Where model endpoints are published. A separate host because the edge in
    # front of a console and the edge in front of an API want opposite things,
    # and because the SDK may need it before it has asked the platform for an
    # endpoint -- the platform's own answer always wins when there is one.
    inference_url: str = DEFAULT_INFERENCE_URL
    timeout: int = 30
    verify_ssl: bool = True

    # Set when auth_token came from the platform credential file. The file is
    # rewritten while the notebook runs, so a long-lived client re-reads it
    # rather than trusting the value it started with.
    auth_token_file: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        load_dotenv()
        verify_ssl_str = os.getenv("CORERUN_VERIFY_SSL", "true").lower()
        return cls(
            auth_token=os.getenv("CORERUN_AUTH_TOKEN"),
            workspace=os.getenv("CORERUN_WORKSPACE"),
            api_url=os.getenv("CORERUN_API_URL", DEFAULT_API_URL),
            inference_url=os.getenv("CORERUN_INFERENCE_URL", DEFAULT_INFERENCE_URL),
            timeout=int(os.getenv("CORERUN_TIMEOUT", "30")),
            verify_ssl=verify_ssl_str not in ("false", "0", "no"),
        )

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from config file"""
        if path is None:
            path = Path.home() / ".corerun" / "config"

        if not path.exists():
            return cls()

        config = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip().strip('"').strip("'")

                    if key == "auth_token":
                        config.auth_token = value
                    elif key == "refresh_token":
                        config.refresh_token = value
                    elif key == "workspace":
                        config.workspace = value
                    elif key == "api_url":
                        config.api_url = value
                    elif key == "inference_url":
                        config.inference_url = value
                    elif key == "timeout":
                        config.timeout = int(value)
                    elif key == "verify_ssl":
                        config.verify_ssl = value.lower() not in ("false", "0", "no")

        return config

    def save(self, path: Optional[Path] = None):
        """Save configuration to file"""
        if path is None:
            path = Path.home() / ".corerun" / "config"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            if self.auth_token:
                f.write(f"auth_token={self.auth_token}\n")
            if self.refresh_token:
                f.write(f"refresh_token={self.refresh_token}\n")
            if self.workspace:
                f.write(f"workspace={self.workspace}\n")
            f.write(f"api_url={self.api_url}\n")
            f.write(f"inference_url={self.inference_url}\n")
            f.write(f"timeout={self.timeout}\n")
            if not self.verify_ssl:
                f.write("verify_ssl=false\n")


def _apply_platform_credential(config: "Config") -> None:
    """
    Fall back to the credential a corerun notebook was launched with.

    Only when nothing else supplied one, so a user who signed in keeps their own
    identity. The path is remembered either way, so the client can pick up a
    renewed key without being rebuilt.
    """
    if config.auth_token:
        return
    key = read_platform_credential()
    if key:
        config.auth_token = key
        config.auth_token_file = credential_path()


# Global configuration
_config: Optional[Config] = None
_client: Optional["CoreRunClient"] = None


def init(
    auth_token: Optional[str] = None,
    workspace: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> "CoreRunClient":
    """
    Initialize the corerun SDK.

    Args:
        auth_token: corerun auth token (or set CORERUN_AUTH_TOKEN env var)
        workspace: Default workspace ID (or set CORERUN_WORKSPACE env var)
        api_url: API base URL (default: https://corerun.ai/api/v1)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        CoreRunClient instance

    Example:
        import corerun
        corerun.init(auth_token="cr-xxx", workspace="my-workspace-id")
    """
    global _config, _client

    # Load from file first, then overlay env vars
    _config = Config.from_file()
    env_config = Config.from_env()
    if env_config.auth_token:
        _config.auth_token = env_config.auth_token
    if env_config.workspace:
        _config.workspace = env_config.workspace
    if os.getenv("CORERUN_API_URL"):
        _config.api_url = env_config.api_url
    if os.getenv("CORERUN_INFERENCE_URL"):
        _config.inference_url = env_config.inference_url
    if os.getenv("CORERUN_TIMEOUT"):
        _config.timeout = env_config.timeout
    # Read from the environment like everything else above it. Parsed in
    # from_env since it was written and dropped here, so CORERUN_VERIFY_SSL did
    # nothing at all unless it was also in the config file -- which meant every
    # development cluster with a self-signed certificate was unreachable with no
    # way to say otherwise.
    if os.getenv("CORERUN_VERIFY_SSL"):
        _config.verify_ssl = env_config.verify_ssl
    _apply_platform_credential(_config)

    # Override with explicit arguments
    if auth_token:
        _config.auth_token = auth_token
    if workspace:
        _config.workspace = workspace
    if api_url:
        _config.api_url = api_url
    if timeout:
        _config.timeout = timeout

    # Validate
    if not _config.auth_token:
        raise ValueError(
            "auth token required. Run `corerun login`, set the CORERUN_AUTH_TOKEN "
            "environment variable, or pass auth_token to corerun.init()"
        )

    # Create client
    from corerun.client import CoreRunClient
    _client = CoreRunClient(_config)

    return _client


def get_config() -> Config:
    """Get current configuration"""
    global _config
    if _config is None:
        # Start with file config (from ~/.corerun/config), then overlay env vars
        _config = Config.from_file()
        env_config = Config.from_env()
        if env_config.auth_token:
            _config.auth_token = env_config.auth_token
        if env_config.workspace:
            _config.workspace = env_config.workspace
        if os.getenv("CORERUN_API_URL"):
            _config.api_url = env_config.api_url
        if os.getenv("CORERUN_INFERENCE_URL"):
            _config.inference_url = env_config.inference_url
        if os.getenv("CORERUN_TIMEOUT"):
            _config.timeout = env_config.timeout
        if os.getenv("CORERUN_VERIFY_SSL"):
            _config.verify_ssl = env_config.verify_ssl
        _apply_platform_credential(_config)
    return _config


def get_client() -> "CoreRunClient":
    """
    Get the current corerun client.

    Raises ValueError if not initialized.
    """
    global _client
    if _client is None:
        # Try auto-init from environment
        config = get_config()
        if config.auth_token:
            return init()
        raise ValueError(
            "corerun SDK not initialized. Call corerun.init() first "
            "or set CORERUN_AUTH_TOKEN environment variable."
        )
    return _client
