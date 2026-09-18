"""
What the GenAI commands put on the wire, and what they make of what comes back.

The Go service is tested where it lives. What cannot be tested there is the
half this file covers: the SDK and the service have to agree on the path, on
the filter names, and on the shape of a reply -- and a disagreement about any
of those reads as "no traces" rather than as a bug.
"""

import re

import httpx
import pytest
from typer.testing import CliRunner

from corerun import config as config_module
from corerun import genai
from corerun.cli import app
from corerun.cli import genai as genai_cli
from corerun.client import CoreRunClient

runner = CliRunner()

BASE = "https://example.test/api/v1"

TRACE = {
    "trace_id": "4536c69e4220348fc027e1292a04da8a",
    "root_span_name": "invoke_agent",
    "state": "OK",
    "duration_ms": 26162.13,
    "total_tokens": 27673,
    "session_id": "20260918_29",
}

DETAIL = {
    **TRACE,
    "spans": [
        {"span_id": "aa", "name": "invoke_agent", "duration_ms": 26162.13, "total_tokens": 27673},
        {"span_id": "bb", "parent_span_id": "aa", "name": "router_call", "duration_ms": 4760.0},
        {"span_id": "cc", "parent_span_id": "bb", "name": "complete", "duration_ms": 4760.0},
    ],
}


@pytest.fixture
def wire(monkeypatch):
    """Answer every request locally, and hand back what was asked for."""
    state = {"seen": [], "respond": None}

    def handle(request: httpx.Request) -> httpx.Response:
        state["seen"].append(request)
        if state["respond"] is not None:
            return state["respond"](request)
        path = request.url.path
        if path.endswith("/genai/sessions"):
            return httpx.Response(200, json={"sessions": [
                {"session_id": "20260918_29", "trace_count": 3, "error_count": 1, "total_tokens": 60184}
            ]})
        if path.endswith("/genai/settings"):
            return httpx.Response(200, json={"retention_days": 30, "sample_rate": 0.25})
        if "/genai/traces/" in path:
            if request.method == "DELETE":
                return httpx.Response(200, json={})
            return httpx.Response(200, json=DETAIL)
        return httpx.Response(200, json={"traces": [TRACE]})

    client = CoreRunClient(config_module.Config(api_url=BASE))
    client._client = httpx.Client(transport=httpx.MockTransport(handle), base_url=BASE)
    monkeypatch.setattr(genai_cli, "_init_client", lambda: None)
    monkeypatch.setattr(config_module, "_client", client)
    return state


def test_traces_go_to_the_genai_prefix_not_the_endpoint_one(wire):
    """/genai/traces and /traces are different features.

    They were deliberately given separate prefixes: one is an agent's span
    tree, the other is a request through a model endpoint. Sending either to
    the other's path returns a plausible, wrong answer.
    """
    genai.traces()
    assert wire["seen"][0].url.path == "/api/v1/genai/traces"


def test_filters_reach_the_wire_under_the_names_the_service_reads(wire):
    genai.traces(state="ERROR", session_id="s-1", model="atlas-galaxy", search="email", limit=10)
    params = dict(wire["seen"][0].url.params)
    assert params["state"] == "ERROR"
    assert params["session_id"] == "s-1"
    assert params["model"] == "atlas-galaxy"
    assert params["search"] == "email"
    assert params["limit"] == "10"


def test_a_filter_left_out_is_not_sent_as_empty(wire):
    """An empty filter must be absent, not present and blank.

    state="" is a request for traces whose state is the empty string, which is
    none of them -- so sending it would turn "no filter" into "no results".
    """
    genai.traces(limit=5)
    params = dict(wire["seen"][0].url.params)
    assert "state" not in params
    assert "session_id" not in params
    assert params["limit"] == "5"


def test_a_trace_comes_back_with_its_spans_typed(wire):
    detail = genai.trace("4536c69e4220348fc027e1292a04da8a")
    assert detail.trace_id == TRACE["trace_id"]
    assert [s.name for s in detail.spans] == ["invoke_agent", "router_call", "complete"]
    assert detail.spans[1].parent_span_id == "aa"


def test_a_field_the_sdk_has_not_heard_of_does_not_break_a_read(wire):
    """The service adds fields between releases.

    An SDK older than the server should return the trace rather than raise on
    a name it does not declare, because the alternative is that every deploy
    breaks every client that has not been upgraded yet.
    """
    wire["respond"] = lambda r: httpx.Response(
        200, json={"traces": [{**TRACE, "a_field_from_the_future": 1}]}
    )
    found = genai.traces()
    assert found[0].trace_id == TRACE["trace_id"]


def test_iter_traces_follows_pages_and_stops(wire):
    """Paging stops on a missing token, and does not re-ask forever."""
    pages = [
        {"traces": [TRACE], "next_page_token": "second"},
        {"traces": [{**TRACE, "trace_id": "b" * 32}]},
    ]
    wire["respond"] = lambda r: httpx.Response(200, json=pages[min(len(wire["seen"]) - 1, 1)])
    got = list(genai.iter_traces(page_size=1))
    assert [t.trace_id for t in got] == [TRACE["trace_id"], "b" * 32]
    assert dict(wire["seen"][1].url.params)["page_token"] == "second"


def test_failed_reads_the_state_rather_than_guessing(wire):
    wire["respond"] = lambda r: httpx.Response(200, json={"traces": [{**TRACE, "state": "ERROR"}]})
    assert genai.traces()[0].failed is True


def test_configure_sends_only_what_was_asked_to_change(wire):
    """Sending sample_rate unasked would overwrite a rate somebody set."""
    genai.configure(retention_days=7)
    body = wire["seen"][0].content.decode()
    assert '"retention_days": 7' in body or '"retention_days":7' in body
    assert "sample_rate" not in body


def test_cli_lists_traces(wire):
    result = runner.invoke(app, ["genai", "traces", "list", "--state", "ERROR"])
    assert result.exit_code == 0, result.output
    assert "invoke_agent" in result.output
    assert dict(wire["seen"][0].url.params)["state"] == "ERROR"


def test_cli_prints_the_span_tree_indented(wire):
    result = runner.invoke(app, ["genai", "traces", "get", "4536c69e"])
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.splitlines() if "router_call" in l or "complete" in l]
    # A child is indented further than its parent, which is the whole point of
    # printing a tree rather than a list.
    assert lines[0].index("router_call") < lines[1].index("complete")


def test_cli_sessions_list(wire):
    result = runner.invoke(app, ["genai", "sessions", "list"])
    assert result.exit_code == 0, result.output
    assert "20260918_29" in result.output


def test_cli_settings_shows_without_changing(wire):
    result = runner.invoke(app, ["genai", "settings"])
    assert result.exit_code == 0, result.output
    assert wire["seen"][0].method == "GET"
    assert "30 days" in result.output


def test_a_missing_trace_reads_as_one_line_not_a_traceback(wire):
    """A trace id that has aged out of retention is an ordinary answer.

    Before this, the command raised through typer and printed a stack trace
    with the useful sentence at the bottom of it.
    """
    wire["respond"] = lambda r: httpx.Response(404, json={"error": "no trace with that id"})
    result = runner.invoke(app, ["genai", "traces", "get", "0" * 32])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "no trace with that id" in result.output


def test_a_failure_is_reported_in_json_mode_too(wire):
    """--json callers get an error document, not prose on stderr."""
    wire["respond"] = lambda r: httpx.Response(404, json={"error": "no trace with that id"})
    result = runner.invoke(app, ["--json", "genai", "traces", "get", "0" * 32])
    assert result.exit_code == 1
    assert '"error"' in result.output


def test_the_id_the_table_prints_is_one_get_accepts(wire):
    """Whatever the table prints must work when pasted back.

    It printed sixteen characters and `get` demanded thirty-two, so copying an
    id out of the listing answered "no trace with that id". Either the column
    prints the whole thing or a prefix resolves; this pins the round trip
    rather than the width, so both remain valid ways to keep it true.
    """
    listing = runner.invoke(app, ["genai", "traces", "list"])
    assert listing.exit_code == 0, listing.output

    # The id as a reader would copy it: the longest unbroken run of hex in the
    # Trace column.
    shown = max(re.findall(r"[0-9a-f]{8,}", listing.output), key=len)
    assert TRACE["trace_id"].startswith(shown)

    wire["seen"].clear()
    fetched = runner.invoke(app, ["genai", "traces", "get", shown])
    assert fetched.exit_code == 0, fetched.output
    assert wire["seen"][-1].url.path == f"/api/v1/genai/traces/{TRACE['trace_id']}"


def test_a_unique_prefix_resolves_to_the_whole_id(wire):
    """Every screen shortens an id, so a reader is always holding a prefix."""
    result = runner.invoke(app, ["genai", "traces", "get", TRACE["trace_id"][:12]])
    assert result.exit_code == 0, result.output
    # The listing is asked for that prefix, then the full id is fetched.
    assert wire["seen"][0].url.path == "/api/v1/genai/traces"
    assert dict(wire["seen"][0].url.params)["search"] == TRACE["trace_id"][:12]
    assert wire["seen"][1].url.path == f"/api/v1/genai/traces/{TRACE['trace_id']}"


def test_a_full_id_is_not_looked_up_first(wire):
    """A whole id costs no extra request."""
    runner.invoke(app, ["genai", "traces", "get", TRACE["trace_id"]])
    assert len(wire["seen"]) == 1
    assert wire["seen"][0].url.path == f"/api/v1/genai/traces/{TRACE['trace_id']}"


def test_an_ambiguous_prefix_says_so_rather_than_picking(wire):
    wire["respond"] = lambda r: httpx.Response(200, json={"traces": [
        {**TRACE, "trace_id": "abc" + "0" * 29},
        {**TRACE, "trace_id": "abc" + "1" * 29},
    ]})
    result = runner.invoke(app, ["genai", "traces", "get", "abc"])
    assert result.exit_code == 1
    assert "names 2 traces" in result.output


def test_a_prefix_matching_nothing_says_which_prefix(wire):
    wire["respond"] = lambda r: httpx.Response(200, json={"traces": []})
    result = runner.invoke(app, ["genai", "traces", "get", "zzzz"])
    assert result.exit_code == 1
    assert "zzzz" in result.output


def test_the_tree_draws_guides_and_a_bar_per_span(wire):
    """The shape of the call is the point, not a list of names."""
    result = runner.invoke(app, ["genai", "traces", "get", TRACE["trace_id"]])
    assert result.exit_code == 0, result.output
    assert "├─" in result.output or "└─" in result.output
    assert "█" in result.output


def test_a_root_child_is_not_indented_under_a_guide_that_was_not_drawn(wire):
    """A root draws no joint, so its children start at the left edge.

    They inherited three columns of padding from a guide that was never
    printed, which read as the whole tree hanging off nothing.
    """
    result = runner.invoke(app, ["genai", "traces", "get", TRACE["trace_id"]])
    joints = [l for l in result.output.splitlines() if l.lstrip().startswith(("├─", "└─"))]
    assert joints, result.output
    # Every line opens with a one-character column that carries "!" for a
    # failed span, so the first joint sits immediately after it -- not three
    # columns further in, which is what inheriting an undrawn guide looked
    # like.
    first = joints[0]
    at = min(i for i in (first.find("├─"), first.find("└─")) if i >= 0)
    assert at <= 1, repr(first)


def test_a_span_too_short_to_fill_a_cell_is_still_drawn(wire):
    """A 0ms span happened. Rounding it away would hide a step."""
    from corerun.cli.genai import _bar

    drawn = _bar(0.5, 0.5, 20)
    assert drawn.strip() != ""


def test_a_bar_sits_where_the_span_ran_not_at_the_left(wire):
    """Position carries as much as length: overlap is the question a slow
    trace raises, and every bar starting at zero cannot answer it."""
    from corerun.cli.genai import _bar

    late = _bar(0.5, 1.0, 20)
    assert late.startswith(" ")
    assert late.index("█") >= 9
