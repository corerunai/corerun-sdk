"""
An environment override is for the process, not for the file.

Exporting CORERUN_API_URL to point one command at another deployment used to be
written back by the next command that saved anything -- so a temporary override
became permanent, and the symptom appeared much later somewhere unrelated: a
sign-in opening the wrong host long after the export was forgotten.
"""

from corerun.config import Config


def test_env_sourced_api_url_is_not_written_back(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text("api_url=https://real.example.com/api/v1\nworkspace=ws-1\ntimeout=30\n")

    config = Config.from_file(cfg)
    # What get_config() does when CORERUN_API_URL is set.
    config.api_url = "http://127.0.0.1:9999/api/v1"
    config.env_only.add("api_url")

    # Something the command legitimately changes.
    config.workspace = "ws-2"
    config.save(cfg)

    written = cfg.read_text()
    assert "api_url=https://real.example.com/api/v1" in written
    assert "127.0.0.1:9999" not in written
    assert "workspace=ws-2" in written


def test_a_setting_the_user_changed_is_written_back(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text("api_url=https://real.example.com/api/v1\n")

    config = Config.from_file(cfg)
    config.api_url = "https://chosen.example.com/api/v1"  # no env_only marker
    config.save(cfg)

    assert "api_url=https://chosen.example.com/api/v1" in cfg.read_text()


def test_env_sourced_workspace_is_not_written_back(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text("api_url=https://real.example.com/api/v1\nworkspace=ws-1\n")

    config = Config.from_file(cfg)
    config.workspace = "ws-from-env"
    config.env_only.add("workspace")
    config.save(cfg)

    assert "workspace=ws-1" in cfg.read_text()
