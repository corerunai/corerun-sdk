"""
The platform credential a corerun notebook is launched with.

These pin the two properties the rest of the design leans on: the key is read
from the file every time rather than remembered, and it never displaces a
credential the user chose themselves.
"""

import pytest

from corerun import config as config_module
from corerun.credentials import credential_path, read_platform_credential


@pytest.fixture
def credential_file(tmp_path, monkeypatch):
    path = tmp_path / "api-key"
    monkeypatch.setenv("CORERUN_AUTH_TOKEN_FILE", str(path))
    return path


def test_absent_credential_is_not_an_error(credential_file):
    # The normal case outside a corerun notebook.
    assert read_platform_credential() is None


def test_reads_the_key_and_strips_the_newline(credential_file):
    credential_file.write_text("cr-abcd1234-secret\n")
    assert read_platform_credential() == "cr-abcd1234-secret"


def test_empty_file_reads_as_absent(credential_file):
    # A half-written file must not be handed out as a key.
    credential_file.write_text("")
    assert read_platform_credential() is None


def test_rereads_after_renewal(credential_file):
    credential_file.write_text("cr-first-secret")
    assert read_platform_credential() == "cr-first-secret"

    # The platform replaces the file while the notebook runs.
    credential_file.write_text("cr-second-secret")
    assert read_platform_credential() == "cr-second-secret"


def test_path_follows_the_env_override(credential_file):
    assert credential_path() == credential_file


def test_platform_credential_fills_an_empty_config(credential_file):
    credential_file.write_text("cr-platform-secret")

    config = config_module.Config()
    config_module._apply_platform_credential(config)

    assert config.auth_token == "cr-platform-secret"
    # Recorded so the client can pick up a renewed key without being rebuilt.
    assert config.auth_token_file == credential_file


def test_platform_credential_does_not_displace_a_chosen_one(credential_file):
    credential_file.write_text("cr-platform-secret")

    config = config_module.Config(auth_token="cr-mine-secret")
    config_module._apply_platform_credential(config)

    assert config.auth_token == "cr-mine-secret"
    # No file to re-read, so a 401 on this key is a real failure.
    assert config.auth_token_file is None


def test_inference_url_survives_both_merges(tmp_path, monkeypatch):
    """A setting parsed from the environment and dropped in the merge is
    invisible: it looks configured and behaves as the default. CORERUN_VERIFY_SSL
    did exactly that for a while, so this pins the new one."""
    import importlib
    import corerun.config as config

    monkeypatch.setenv("CORERUN_INFERENCE_URL", "https://api-dev.corerun.ai")
    monkeypatch.setenv("HOME", str(tmp_path))
    importlib.reload(config)

    assert config.get_config().inference_url == "https://api-dev.corerun.ai"


def test_inference_url_defaults_and_is_overridable_by_file(tmp_path, monkeypatch):
    import importlib
    import corerun.config as config

    monkeypatch.delenv("CORERUN_INFERENCE_URL", raising=False)
    importlib.reload(config)
    assert config.DEFAULT_INFERENCE_URL == "https://api.corerun.ai"

    path = tmp_path / "config"
    path.write_text("inference_url=https://inference.example.internal\n")
    loaded = config.Config.from_file(path)
    assert loaded.inference_url == "https://inference.example.internal"
