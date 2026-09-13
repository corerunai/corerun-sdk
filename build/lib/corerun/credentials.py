"""
Platform credential discovery.

A notebook launched by corerun is given a credential to call the platform back
with. It arrives as a file rather than an environment variable because it is
short-lived: the platform replaces it every few hours, for as long as the
notebook is running, and a container's environment cannot be rewritten while a
process is using it. An abandoned notebook stops being renewed and its
credential expires on its own.

The practical consequence for callers is that the key must be read when it is
needed, not remembered from process start.
"""

import os
from pathlib import Path
from typing import Optional

# Where the platform installs the credential. Matches drivers.AgentCredentialFile
# on the Go side; the env var exists so a test or an unusual image can move it.
DEFAULT_CREDENTIAL_FILE = "/etc/corerun/api-key"


def credential_path() -> Path:
    """Path the platform credential is expected at."""
    return Path(os.getenv("CORERUN_AUTH_TOKEN_FILE", DEFAULT_CREDENTIAL_FILE))


def read_platform_credential() -> Optional[str]:
    """
    Read the credential the platform installed, or None if there isn't one.

    Absence is the normal case outside a corerun notebook, so it is not an
    error. An unreadable file is treated the same way: the caller falls back to
    whatever other credential it has, which produces a clearer failure than an
    exception from a path most users have never heard of.
    """
    try:
        key = credential_path().read_text().strip()
    except OSError:
        return None
    return key or None
