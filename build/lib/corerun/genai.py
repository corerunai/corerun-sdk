"""
GenAI observability for the corerun SDK.

Agent traces: a span tree for one invocation, the sessions those traces group
into, and what a workspace has asked to keep.

These are not the same thing as `corerun.traces`, which is the record of
requests through a model endpoint. Two features, two prefixes, and a token
scoped to one does not reach the other.

Sending traces needs nothing from here: point any OpenTelemetry exporter at
the platform and it arrives.

    OTEL_EXPORTER_OTLP_ENDPOINT=https://<host>
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer <key>,X-Workspace-ID=<id>

This module is for reading them back.

Usage:
    import corerun

    corerun.init(auth_token="...")

    # The most recent traces
    for trace in corerun.genai.traces():
        print(trace.trace_id, trace.root_span_name, trace.duration_ms)

    # One trace, with its spans
    detail = corerun.genai.trace("4536c69e4220348fc027e1292a04da8a")
    for span in detail.spans:
        print(span.name, span.duration_ms)

    # Conversations
    for session in corerun.genai.sessions():
        print(session.session_id, session.trace_count)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from corerun.config import get_client


@dataclass
class Trace:
    """One agent invocation, summarised."""

    trace_id: str
    root_span_name: Optional[str] = None
    state: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    provider: Optional[str] = None
    request_model: Optional[str] = None
    agent_name: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.state == "ERROR"


@dataclass
class Span:
    """One step inside a trace."""

    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    span_type: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    status_message: Optional[str] = None
    operation: Optional[str] = None
    provider: Optional[str] = None
    request_model: Optional[str] = None
    response_model: Optional[str] = None
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    inputs: Any = None
    outputs: Any = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: Any = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    reasoning_output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class TraceDetail(Trace):
    """A trace with the spans it is made of."""

    spans: List[Span] = field(default_factory=list)


@dataclass
class Session:
    """A conversation: the traces sharing one session id."""

    session_id: str
    trace_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class Settings:
    """What this workspace keeps, and how much of it it records."""

    retention_days: Optional[int] = None
    sample_rate: float = 1.0
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None


def _known(cls, payload: Dict[str, Any]):
    """Build a dataclass from a payload, ignoring fields it does not declare.

    The service adds fields between releases, and an SDK older than the server
    should return the trace rather than raise on a name it has not heard of.
    """
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in payload.items() if k in allowed})


def traces(
    *,
    search: Optional[str] = None,
    state: Optional[str] = None,
    session_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    model: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 50,
    page_token: Optional[str] = None,
    workspace: Optional[str] = None,
) -> List[Trace]:
    """One page of traces, most recent first.

    Args:
        search: matches trace id, input or output
        state: OK, ERROR or IN_PROGRESS
        session_id: only the traces in one conversation
        experiment_id: only the traces of one experiment
        model: only traces that asked this model
        since / until: RFC 3339 timestamps
        limit: how many to return
        page_token: the token from a previous page
        workspace: workspace id (uses the default if not given)

    Example:
        failed = corerun.genai.traces(state="ERROR", limit=10)
    """
    params = {
        "search": search,
        "state": state,
        "session_id": session_id,
        "experiment_id": experiment_id,
        "model": model,
        "since": since,
        "until": until,
        "limit": limit,
        "page_token": page_token,
    }
    response = get_client().get(
        "/genai/traces",
        params={k: v for k, v in params.items() if v is not None},
        workspace=workspace,
    )
    return [_known(Trace, t) for t in response.get("traces") or []]


def iter_traces(*, page_size: int = 100, workspace: Optional[str] = None, **filters) -> Iterator[Trace]:
    """Every trace matching the filters, following the pages.

    A generator rather than a list: a busy workspace has more traces than
    anyone wants in memory, and most callers stop early.

    Example:
        for trace in corerun.genai.iter_traces(state="ERROR"):
            print(trace.trace_id)
    """
    token: Optional[str] = None
    while True:
        params = {
            **{k: v for k, v in filters.items() if v is not None},
            "limit": page_size,
        }
        if token:
            params["page_token"] = token
        response = get_client().get("/genai/traces", params=params, workspace=workspace)
        page = response.get("traces") or []
        for payload in page:
            yield _known(Trace, payload)
        token = response.get("next_page_token")
        if not token or not page:
            return


def trace(trace_id: str, *, workspace: Optional[str] = None) -> TraceDetail:
    """One trace and its spans.

    Example:
        detail = corerun.genai.trace("4536c69e...")
        print(len(detail.spans), "spans")
    """
    payload = get_client().get(f"/genai/traces/{trace_id}", workspace=workspace)
    detail = _known(TraceDetail, payload)
    detail.spans = [_known(Span, s) for s in payload.get("spans") or []]
    return detail


def delete_trace(trace_id: str, *, workspace: Optional[str] = None) -> None:
    """Delete a trace and everything recorded about it."""
    get_client().delete(f"/genai/traces/{trace_id}", workspace=workspace)


def sessions(
    *,
    limit: int = 50,
    page_token: Optional[str] = None,
    workspace: Optional[str] = None,
) -> List[Session]:
    """Conversations, most recently active first.

    Example:
        for s in corerun.genai.sessions():
            print(s.session_id, s.trace_count, "traces")
    """
    params: Dict[str, Any] = {"limit": limit}
    if page_token:
        params["page_token"] = page_token
    response = get_client().get("/genai/sessions", params=params, workspace=workspace)
    return [_known(Session, s) for s in response.get("sessions") or []]


def settings(*, workspace: Optional[str] = None) -> Settings:
    """What this workspace keeps, and how much of it it records."""
    return _known(Settings, get_client().get("/genai/settings", workspace=workspace))


def configure(
    *,
    retention_days: Optional[int] = None,
    sample_rate: Optional[float] = None,
    workspace: Optional[str] = None,
) -> Settings:
    """Change retention or sampling.

    Sampling is not a performance knob for the platform; it is how much of what
    an application sends is kept. Retention is in days, and None means keep
    until something deletes it.

    Example:
        corerun.genai.configure(retention_days=30, sample_rate=0.25)
    """
    body: Dict[str, Any] = {}
    if retention_days is not None:
        body["retention_days"] = retention_days
    if sample_rate is not None:
        body["sample_rate"] = sample_rate
    return _known(Settings, get_client().put("/genai/settings", json=body, workspace=workspace))
