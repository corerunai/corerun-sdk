"""
What the accelerator commands actually put on the wire.

The Go side is tested where it lives. What cannot be tested there is the half
this file covers: the CLI and the API have to agree on how a family name is
addressed, and they agree in exactly one place -- the path. A family is named
`nvidia/hopper`, a slash is a path separator, and a name that arrives as two
segments matches no route at all. That failure reads as "no such family" rather
than as a bug, so it is worth pinning rather than reasoning about.
"""

import json

import httpx
import pytest
from typer.testing import CliRunner

from corerun import config as config_module
from corerun.cli import accelerators as accelerators_cli
from corerun.cli import app
from corerun.client import CoreRunClient

runner = CliRunner()

BASE = "https://example.test/api/v1"


@pytest.fixture
def wire(monkeypatch):
    """Answer every request locally, and hand back what was asked for."""
    state = {"seen": [], "respond": None}

    def handle(request: httpx.Request) -> httpx.Response:
        state["seen"].append(request)
        if state["respond"] is not None:
            return state["respond"](request)
        if request.method == "DELETE":
            return httpx.Response(200, json={"message": "Removed"})
        return httpx.Response(200, json={"Name": "nvidia/rubin", "Card": "rtx pro 5000"})

    # The real client, with its transport swapped underneath -- so what the
    # assertions see is the request the CLI really builds.
    client = CoreRunClient(config_module.Config(api_url=BASE))
    client._client = httpx.Client(transport=httpx.MockTransport(handle), base_url=BASE)

    monkeypatch.setattr(accelerators_cli, "_init_client", lambda: None)
    monkeypatch.setattr(config_module, "_client", client)
    return state


def test_a_family_with_a_slash_is_one_path_segment(wire, tmp_path):
    doc = tmp_path / "rubin.yaml"
    doc.write_text("display_name: Rubin\n")

    result = runner.invoke(app, ["accelerators", "add", "nvidia/rubin", "--from-file", str(doc)])
    assert result.exit_code == 0, result.output

    request = wire["seen"][0]
    assert request.method == "PUT"
    # The escape, spelled out: one segment, not two.
    assert request.url.raw_path == b"/api/v1/admin/accelerators/families/nvidia%2Frubin"
    # And the bytes a person wrote, labelled as what they are rather than
    # relabelled as JSON.
    assert request.content == b"display_name: Rubin\n"
    assert request.headers["content-type"].startswith("application/yaml")


def test_an_inline_document_is_sent_as_json(wire):
    result = runner.invoke(
        app, ["accelerators", "add", "nvidia/rubin", "--json", '{"display_name":"Rubin"}']
    )
    assert result.exit_code == 0, result.output
    assert wire["seen"][0].content == b'{"display_name":"Rubin"}'
    assert wire["seen"][0].headers["content-type"] == "application/json"


def test_a_family_name_without_a_slash_is_left_alone(wire):
    # `cpu` is a family too, and escaping it would be cargo cult.
    result = runner.invoke(app, ["accelerators", "rm", "cpu", "--yes"])
    assert result.exit_code == 0, result.output
    assert wire["seen"][0].url.raw_path == b"/api/v1/admin/accelerators/families/cpu"


def test_tagging_sends_the_family_and_the_device_id(wire):
    result = runner.invoke(
        app,
        ["accelerators", "tag", "RTX PRO 5000", "--family", "nvidia/blackwell-rtx",
         "--device-id", "10de:2bb1"],
    )
    assert result.exit_code == 0, result.output

    body = json.loads(wire["seen"][0].content)
    assert body == {
        "card": "RTX PRO 5000",
        "device_id": "10de:2bb1",
        "family": "nvidia/blackwell-rtx",
        # Sent empty rather than omitted, and the store reads an empty one as
        # "leave the note alone" -- so re-tagging a card does not erase what
        # somebody wrote about it.
        "note": "",
    }


def test_showing_a_family_by_its_full_name(wire):
    # The name `accelerators list` prints. Copying it out of that output has to
    # work, or the listing is inviting a 404.
    result = runner.invoke(app, ["accelerators", "show", "nvidia/hopper"])
    assert result.exit_code == 0, result.output
    assert wire["seen"][0].url.raw_path == b"/api/v1/inference-servers/accelerators/nvidia%2Fhopper"


def test_an_api_that_predates_the_route_is_not_read_as_no_tags(wire):
    # An older API answers the single-accelerator route for "cards", and an
    # unclassified accelerator is a normal answer. Reporting that as "nothing
    # is tagged" would be a lie somebody acts on by tagging what is already
    # tagged.
    wire["respond"] = lambda _: httpx.Response(200, json={"token": "cards", "known": False})

    result = runner.invoke(app, ["accelerators", "cards"])
    assert result.exit_code == 1
    assert "does not serve card tags" in result.output
