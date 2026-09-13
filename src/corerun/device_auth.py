"""
Device authorization for shells that cannot open a browser.

A notebook terminal or an SSH session has no way to complete a redirect-based
sign-in. Instead the CLI asks the platform for a grant, shows a short code, and
waits while the user approves it from a browser where they are already signed
in. The key that comes back belongs to the approving user, so the shell can do
what that person can and no more.
"""

import time
from typing import Callable, Optional

import httpx

from corerun.config import DEFAULT_API_URL  # noqa: F401

# How long to keep polling before giving up, independent of what the server
# says: a terminal left waiting forever is worse than one that stops.
MAX_WAIT_SECONDS = 600


class DeviceAuthError(Exception):
    """Device authorization could not be completed."""


def start(api_url: str, client_name: str, verify_ssl: bool = True) -> dict:
    """
    Ask for a device grant.

    Returns the payload holding device_code, user_code and verification_uri.
    """
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/auth/device",
            json={"client_name": client_name},
            timeout=30,
            verify=verify_ssl,
        )
    except httpx.HTTPError as exc:
        raise DeviceAuthError(f"Could not reach {api_url}: {exc}") from exc

    if response.status_code != 200:
        raise DeviceAuthError(_error_message(response, "Could not start sign-in"))
    return response.json()


def poll(
    api_url: str,
    device_code: str,
    interval: int = 5,
    verify_ssl: bool = True,
    on_wait: Optional[Callable[[], None]] = None,
) -> dict:
    """
    Wait for the grant to be approved and exchange it for a key.

    Returns the payload holding auth_token and workspace_id. Raises DeviceAuthError
    if the request is denied, expires, or the wait runs out.
    """
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    url = f"{api_url.rstrip('/')}/auth/device/token"

    while time.monotonic() < deadline:
        try:
            response = httpx.post(
                url, json={"device_code": device_code}, timeout=30, verify=verify_ssl
            )
        except httpx.HTTPError as exc:
            raise DeviceAuthError(f"Could not reach {api_url}: {exc}") from exc

        if response.status_code == 200:
            return response.json()

        # Still waiting on the person, which is the expected case for most of
        # this loop rather than a failure.
        if response.status_code == 202:
            body = response.json()
            interval = body.get("interval", interval) or interval
            if on_wait:
                on_wait()
            time.sleep(interval)
            continue

        raise DeviceAuthError(_error_message(response, "Sign-in failed"))

    raise DeviceAuthError("Timed out waiting for approval")


def _error_message(response: httpx.Response, fallback: str) -> str:
    try:
        body = response.json()
    except Exception:
        return f"{fallback} (HTTP {response.status_code})"
    return body.get("message") or body.get("error") or f"{fallback} (HTTP {response.status_code})"
