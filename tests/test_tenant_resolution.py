"""Where the CLI finds the tenant a model's repository belongs to.

`corerun models push` builds the repository URL out of it, so a response
without it is not a degraded push -- it is no push at all, failing before
anything is uploaded with "Could not determine your tenant". /auth/me returned
exactly that for a while: the user, and nothing about which tenant they are in.

Both shapes are accepted because both are sent: the key at the top level and
the same key inside `user`.
"""
import pytest

from corerun.cli import registry


class _Client:
    def __init__(self, response):
        self.response = response
        self.asked = None

    def get(self, path, workspace=None):
        self.asked = (path, workspace)
        return self.response


@pytest.fixture
def client(monkeypatch):
    def install(response):
        c = _Client(response)
        monkeypatch.setattr("corerun.config.get_client", lambda: c)
        return c
    return install


TENANT = "e3f4c81c-d73b-47b3-b569-11ba961c5944"


def test_top_level(client):
    client({"tenant_id": TENANT, "user": {"id": "u1"}})
    assert registry._tenant_id(None) == TENANT


def test_nested_under_user(client):
    client({"user": {"id": "u1", "tenant_id": TENANT}})
    assert registry._tenant_id(None) == TENANT


def test_the_workspace_is_passed_through(client):
    c = client({"tenant_id": TENANT})
    registry._tenant_id("ws-1")
    assert c.asked == ("/auth/me", "ws-1")


def test_no_tenant_says_so(client):
    # The shape that broke it: a user, and nothing about their tenant.
    client({"needs_onboarding": False, "user": {"id": "u1", "email": "a@b.c"}})
    with pytest.raises(Exception, match="Could not determine your tenant"):
        registry._tenant_id(None)
