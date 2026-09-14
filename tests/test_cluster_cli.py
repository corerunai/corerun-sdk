"""
Adding a cluster, and what the command puts where.

Two halves are easy to get wrong and invisible when they are.

The first is the request: `POST /clusters/prepare` answers YAML for a
Kubernetes cluster and JSON for a host, and the ordinary client path decodes
every success as JSON -- so a manifest cannot come back through it at all. It
also reads `workspace_scope`, and a false there is not a preference but a
different authorization check, which answers 403 rather than making the cluster.

The second is the streams. `corerun cluster add gpu1 > cluster.yaml` has to
leave a file kubectl will accept, so the manifest goes to stdout and every word
of advice goes to stderr. A missing split produces a YAML file with an
apology in it, which fails much later and much further away.
"""

import json

import httpx
import pytest
from typer.testing import CliRunner

from corerun import config as config_module
from corerun.cli import app
from corerun.cli import clusters as clusters_cli
from corerun.client import CoreRunClient

runner = CliRunner()

BASE = "https://example.test/api/v1"

MANIFEST = "# corerun Cluster Agent Deployment\n---\nkind: Namespace\n"


@pytest.fixture
def wire(monkeypatch):
    """Answer every request locally, and hand back what was asked for."""
    state = {"seen": [], "respond": None}

    def handle(request: httpx.Request) -> httpx.Response:
        state["seen"].append(request)
        if state["respond"] is not None:
            return state["respond"](request)

        path = request.url.path
        if path.endswith("/clusters/prepare"):
            body = json.loads(request.content)
            if body.get("backend") == "host":
                return httpx.Response(
                    201,
                    json={
                        "name": body["name"],
                        "backend": "host",
                        "architecture": body.get("architecture", "amd64"),
                        "os": body.get("os", "linux"),
                        "enrollment_token": "ENROLL",
                        "install_command": (
                            "curl -fsSL "
                            "https://example.test/api/v1/connectors/install.sh"
                            " | sudo bash"
                        ),
                        "connect_command": "sudo corerun-host-connector connect --token ENROLL",
                        "note": "Run the install command first, then connect.",
                    },
                )
            return httpx.Response(201, text=MANIFEST, headers={"Content-Type": "text/yaml"})
        if path.endswith("/connector-manifest"):
            return httpx.Response(200, text=MANIFEST, headers={"Content-Type": "text/yaml"})
        if path.endswith("/token"):
            return httpx.Response(200, json={"token": "NEW-TOKEN", "message": "Token regenerated"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"message": "Deleted", "cleanup": []})
        return httpx.Response(200, json={"clusters": []})

    client = CoreRunClient(config_module.Config(api_url=BASE))
    client._client = httpx.Client(transport=httpx.MockTransport(handle), base_url=BASE)

    monkeypatch.setattr(clusters_cli, "_init_client", lambda: None)
    monkeypatch.setattr(config_module, "_client", client)
    return state


def body_of(request) -> dict:
    return json.loads(request.content) if request.content else {}


def test_the_manifest_is_the_only_thing_on_stdout(wire):
    result = runner.invoke(app, ["clusters", "add", "gpu1"])
    assert result.exit_code == 0, result.stderr

    # A redirect must produce a file kubectl accepts: the manifest, complete,
    # and nothing else.
    assert result.stdout == MANIFEST
    assert "kubectl apply" not in result.stdout

    # And the person still gets told what to do with it.
    assert "kubectl apply" in result.stderr
    assert "gpu1" in result.stderr


def test_adding_a_cluster_sends_the_fields_the_api_reads(wire):
    result = runner.invoke(
        app,
        [
            "clusters", "add", "gpu1",
            "--namespace", "agents",
            "--architecture", "arm64",
            "--accelerator-family", "h100",
        ],
    )
    assert result.exit_code == 0, result.stderr

    body = body_of(wire["seen"][0])
    assert body["name"] == "gpu1"
    assert body["backend"] == "kubernetes"
    assert body["namespace"] == "agents"
    assert body["architecture"] == "arm64"
    assert body["accelerator_family"] == "h100"
    # The narrower scope, unless asked otherwise: sending false here sends the
    # request down the tenant-administrator branch, which refuses rather than
    # creating a workspace's cluster.
    assert body["workspace_scope"] is True


def test_tenant_wide_is_the_only_thing_that_clears_the_scope(wire):
    result = runner.invoke(app, ["clusters", "add", "gpu1", "--tenant-wide"])
    assert result.exit_code == 0, result.stderr
    assert body_of(wire["seen"][0])["workspace_scope"] is False


def test_json_mode_keeps_the_manifest_out_of_stdout(wire):
    # Under --json, stdout is a document a script parses. A manifest printed
    # alongside it makes the whole thing unparseable.
    result = runner.invoke(app, ["--json", "clusters", "add", "gpu1"])
    assert result.exit_code == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload["name"] == "gpu1"
    assert payload["manifest"] == MANIFEST


def test_removing_sends_the_cleanup_flags_explicitly(wire):
    # The API only defaults clean_resources to true when it cannot parse a body
    # at all. A body without the field means false -- so a cluster removed from
    # here would leave its Helm release and RBAC behind, silently.
    result = runner.invoke(app, ["clusters", "rm", "gpu1", "--yes"])
    assert result.exit_code == 0, result.stderr

    request = wire["seen"][0]
    assert request.method == "DELETE"
    body = body_of(request)
    assert body["clean_resources"] is True
    assert body["delete_namespace"] is False
    assert body["confirm_name"] == "gpu1"


def test_a_tenant_wide_cluster_explains_why_it_will_not_remove(wire):
    wire["respond"] = lambda _: httpx.Response(
        403, json={"error": "tenant_scoped", "message": "tenant_scoped"}
    )

    result = runner.invoke(app, ["clusters", "rm", "gpu1", "--yes"])
    assert result.exit_code == 1
    # Not just the API's word for it -- what to do instead.
    assert "tenant" in result.stderr.lower()


def test_the_manifest_can_be_fetched_again(wire):
    result = runner.invoke(app, ["clusters", "manifest", "gpu1"])
    assert result.exit_code == 0, result.stderr
    assert result.stdout == MANIFEST
    assert wire["seen"][0].url.path.endswith("/clusters/gpu1/connector-manifest")


def test_rotating_says_the_agent_will_drop_off(wire):
    result = runner.invoke(app, ["clusters", "token", "rotate", "gpu1", "--yes"])
    assert result.exit_code == 0, result.stderr

    assert "NEW-TOKEN" in result.stdout
    # Rotating through the API does not set the previous-token fallback the way
    # the hub's own rotation does, so a connected agent cannot recover by
    # itself. Saying so is the difference between an inconvenience and an
    # outage nobody can explain.
    assert "re-apply" in result.stderr.lower() or "manifest" in result.stderr.lower()


def test_the_global_json_flag_reaches_the_old_commands(wire):
    # These three commands predate the newer style and carry a --json of their
    # own. It only ever turns the mode on, so the global flag survives -- worth
    # pinning, because the two flags meaning the same thing is easy to break
    # and silent when broken.
    result = runner.invoke(app, ["--json", "clusters", "list"])
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == []
