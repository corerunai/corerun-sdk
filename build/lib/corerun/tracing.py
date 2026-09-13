"""
corerun Tracing Module

Provides automatic traceability for inference code and agentic frameworks.
Compatible with MLflow GenAI tracing patterns.

Usage:
    import corerun
    from corerun.tracing import trace, TracingClient

    # Initialize with tracing
    corerun.init(auth_token="...", workspace="...", enable_tracing=True)

    # Decorator-based tracing
    @trace(span_type="CHAT_MODEL")
    def my_llm_call(messages):
        # Your LLM code here
        return response

    # Context manager tracing
    with trace("embedding_step", span_type="EMBEDDING") as span:
        span.set_inputs({"text": "hello"})
        result = embed(text)
        span.set_outputs({"embedding": result})

    # Auto-instrumentation for popular frameworks
    from corerun.tracing import instrument
    instrument("openai")  # Auto-trace OpenAI calls
    instrument("anthropic")  # Auto-trace Anthropic calls
    instrument("langchain")  # Auto-trace LangChain
"""

import os
import time
import uuid
import json
import functools
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import httpx
except ImportError:
    httpx = None


class SpanType(str, Enum):
    """MLflow-compatible span types."""
    CHAT_MODEL = "CHAT_MODEL"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    TOOL = "TOOL"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    PARSER = "PARSER"
    RERANKER = "RERANKER"
    UNKNOWN = "UNKNOWN"


class SpanStatus(str, Enum):
    """Span status values."""
    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


@dataclass
class SpanEvent:
    """An event within a span."""
    name: str
    timestamp: int  # Unix ms
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single span in a trace."""
    span_id: str
    name: str
    span_type: SpanType
    start_time: int  # Unix ms
    parent_id: Optional[str] = None
    end_time: Optional[int] = None
    duration_ms: Optional[int] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None

    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """Set span inputs."""
        self.inputs = inputs

    def set_outputs(self, outputs: Dict[str, Any]) -> None:
        """Set span outputs."""
        self.outputs = outputs

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple span attributes."""
        self.attributes.update(attributes)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=int(time.time() * 1000),
            attributes=attributes or {}
        ))

    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self, status: Optional[SpanStatus] = None) -> None:
        """End the span."""
        self.end_time = int(time.time() * 1000)
        self.duration_ms = self.end_time - self.start_time
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for API submission."""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id or "",
            "name": self.name,
            "span_type": self.span_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time or int(time.time() * 1000),
            "duration_ms": self.duration_ms or 0,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "status": self.status.value,
            "status_message": self.status_message or "",
        }


@dataclass
class Trace:
    """A complete trace containing multiple spans."""
    trace_id: str
    experiment_id: str = ""
    server_id: str = ""
    model: str = ""
    provider: str = ""
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add_span(self, span: Span) -> None:
        """Add a span to the trace."""
        self.spans.append(span)

    @property
    def root_span(self) -> Optional[Span]:
        """Get the root span (no parent)."""
        for span in self.spans:
            if not span.parent_id:
                return span
        return self.spans[0] if self.spans else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for API submission."""
        root = self.root_span
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = 0

        # Extract token info from spans
        for span in self.spans:
            if "usage" in span.outputs:
                usage = span.outputs["usage"]
                total_tokens += usage.get("total_tokens", 0)
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
            if span.duration_ms:
                latency_ms = max(latency_ms, span.duration_ms)

        # Get prompt/response previews
        prompt_preview = ""
        response_preview = ""
        if root:
            if "messages" in root.inputs:
                messages = root.inputs["messages"]
                if messages:
                    last_user = next(
                        (m for m in reversed(messages) if m.get("role") == "user"),
                        None
                    )
                    if last_user:
                        prompt_preview = str(last_user.get("content", ""))[:500]
            if "message" in root.outputs:
                msg = root.outputs["message"]
                response_preview = str(msg.get("content", ""))[:500]

        return {
            "trace_id": self.trace_id,
            "request_id": self.trace_id,
            "experiment_id": self.experiment_id,
            "server_id": self.server_id,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "prompt_preview": prompt_preview,
            "response_preview": response_preview,
            "streaming": False,
            "status": "ok" if all(s.status == SpanStatus.OK for s in self.spans) else "error",
            "spans": [s.to_dict() for s in self.spans],
            "start_time": self.start_time.isoformat() if self.start_time else datetime.utcnow().isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else datetime.utcnow().isoformat(),
        }


class TracingClient:
    """Client for sending traces to corerun API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        workspace_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        auto_flush: bool = True,
        flush_interval: float = 5.0,
    ):
        from corerun.config import DEFAULT_API_URL

        self.api_url = api_url or os.getenv("CORERUN_API_URL", DEFAULT_API_URL)
        self.auth_token = auth_token or os.getenv("CORERUN_AUTH_TOKEN", "")
        self.workspace_id = workspace_id or os.getenv("CORERUN_WORKSPACE", "")
        self.experiment_id = experiment_id or os.getenv("CORERUN_EXPERIMENT", "sdk-traces")
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval

        self._traces: List[Trace] = []
        self._lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flush = threading.Event()

        if auto_flush and self.auth_token:
            self._start_flush_thread()

    def _start_flush_thread(self):
        """Start background flush thread."""
        def flush_loop():
            while not self._stop_flush.wait(self.flush_interval):
                self.flush()

        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()

    def add_trace(self, trace: Trace) -> None:
        """Add a trace to be flushed."""
        with self._lock:
            self._traces.append(trace)

    def flush(self) -> None:
        """Flush all pending traces to the API."""
        if not self.auth_token or not httpx:
            return

        with self._lock:
            traces = self._traces.copy()
            self._traces.clear()

        for trace in traces:
            try:
                self._send_trace(trace)
            except Exception as e:
                print(f"[corerun Tracing] Failed to send trace: {e}")

    def _send_trace(self, trace: Trace) -> None:
        """Send a single trace to the API."""
        if not httpx:
            return

        trace.experiment_id = trace.experiment_id or self.experiment_id

        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{self.api_url}/traces",
                json=trace.to_dict(),
                headers={
                    "Authorization": f"Bearer {self.auth_token}",
                    "X-Workspace-ID": self.workspace_id,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

    def close(self) -> None:
        """Close the client and flush remaining traces."""
        self._stop_flush.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)
        self.flush()


# Global tracing state
_tracing_client: Optional[TracingClient] = None
_current_trace: Optional[Trace] = None
_current_span: Optional[Span] = None
_span_stack: List[Span] = []
_enabled: bool = False


def configure(
    api_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    workspace_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    enabled: bool = True,
) -> None:
    """Configure tracing.

    Args:
        api_url: corerun API URL
        api_key: API key for authentication
        workspace_id: Workspace ID
        experiment_id: Default experiment ID for traces
        enabled: Whether tracing is enabled
    """
    global _tracing_client, _enabled

    _enabled = enabled
    if enabled:
        _tracing_client = TracingClient(
            api_url=api_url,
            auth_token=auth_token,
            workspace_id=workspace_id,
            experiment_id=experiment_id,
        )


def is_enabled() -> bool:
    """Check if tracing is enabled."""
    return _enabled and _tracing_client is not None


def get_current_span() -> Optional[Span]:
    """Get the current active span."""
    return _span_stack[-1] if _span_stack else None


def get_current_trace() -> Optional[Trace]:
    """Get the current active trace."""
    return _current_trace


@contextmanager
def start_trace(
    experiment_id: Optional[str] = None,
    model: str = "",
    provider: str = "",
    server_id: str = "",
):
    """Start a new trace context.

    Usage:
        with start_trace(model="gpt-4") as trace:
            # Your code here
            pass
    """
    global _current_trace

    trace = Trace(
        trace_id=str(uuid.uuid4()),
        experiment_id=experiment_id or "",
        model=model,
        provider=provider,
        server_id=server_id,
        start_time=datetime.utcnow(),
    )

    old_trace = _current_trace
    _current_trace = trace

    try:
        yield trace
    finally:
        trace.end_time = datetime.utcnow()
        _current_trace = old_trace

        if is_enabled() and _tracing_client:
            _tracing_client.add_trace(trace)


@contextmanager
def start_span(
    name: str,
    span_type: Union[SpanType, str] = SpanType.UNKNOWN,
    inputs: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Start a new span context.

    Usage:
        with start_span("llm_call", span_type=SpanType.LLM) as span:
            span.set_inputs({"prompt": "Hello"})
            result = call_llm()
            span.set_outputs({"response": result})
    """
    global _current_trace

    if isinstance(span_type, str):
        span_type = SpanType(span_type) if span_type in SpanType.__members__ else SpanType.UNKNOWN

    parent = get_current_span()

    span = Span(
        span_id=str(uuid.uuid4()),
        name=name,
        span_type=span_type,
        start_time=int(time.time() * 1000),
        parent_id=parent.span_id if parent else None,
        inputs=inputs or {},
        attributes=attributes or {},
    )

    _span_stack.append(span)

    # Auto-create trace if none exists
    auto_trace = False
    if _current_trace is None:
        _current_trace = Trace(
            trace_id=str(uuid.uuid4()),
            start_time=datetime.utcnow(),
        )
        auto_trace = True

    _current_trace.add_span(span)

    try:
        yield span
        span.end(SpanStatus.OK)
    except Exception as e:
        span.set_status(SpanStatus.ERROR, str(e))
        span.add_event("exception", {
            "exception.type": type(e).__name__,
            "exception.message": str(e),
        })
        span.end(SpanStatus.ERROR)
        raise
    finally:
        _span_stack.pop()

        # Auto-flush trace if we auto-created it
        if auto_trace and is_enabled() and _tracing_client:
            _current_trace.end_time = datetime.utcnow()
            _tracing_client.add_trace(_current_trace)
            _current_trace = None


def trace(
    name: Optional[str] = None,
    span_type: Union[SpanType, str] = SpanType.UNKNOWN,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
):
    """Decorator to trace a function.

    Usage:
        @trace(span_type=SpanType.LLM)
        def my_llm_call(messages):
            return response

        @trace("embedding_step", span_type="EMBEDDING")
        def embed(text):
            return embedding
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_enabled():
                return func(*args, **kwargs)

            inputs = {}
            if capture_inputs:
                # Capture positional args
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                for i, arg in enumerate(args):
                    if i < len(params):
                        inputs[params[i]] = _safe_serialize(arg)
                # Capture keyword args
                for k, v in kwargs.items():
                    inputs[k] = _safe_serialize(v)

            with start_span(span_name, span_type=span_type, inputs=inputs) as span:
                result = func(*args, **kwargs)

                if capture_outputs:
                    span.set_outputs({"result": _safe_serialize(result)})

                return result

        return wrapper
    return decorator


def _safe_serialize(obj: Any, max_depth: int = 3) -> Any:
    """Safely serialize an object for tracing."""
    if max_depth <= 0:
        return str(obj)[:500]

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item, max_depth - 1) for item in obj[:100]]

    if isinstance(obj, dict):
        return {
            str(k)[:100]: _safe_serialize(v, max_depth - 1)
            for k, v in list(obj.items())[:50]
        }

    # For other objects, try to get a dict representation
    if hasattr(obj, "__dict__"):
        return _safe_serialize(obj.__dict__, max_depth - 1)

    if hasattr(obj, "model_dump"):  # Pydantic v2
        return _safe_serialize(obj.model_dump(), max_depth - 1)

    if hasattr(obj, "dict"):  # Pydantic v1
        return _safe_serialize(obj.dict(), max_depth - 1)

    return str(obj)[:500]


# ========================
# Auto-instrumentation
# ========================

_instrumented: set = set()


def instrument(library: str) -> None:
    """Auto-instrument a library for tracing.

    Supported libraries:
        - openai: OpenAI Python client
        - anthropic: Anthropic Python client
        - langchain: LangChain framework
        - litellm: LiteLLM proxy
        - llamaindex: LlamaIndex framework

    Usage:
        from corerun.tracing import instrument
        instrument("openai")
    """
    if library in _instrumented:
        return

    if library == "openai":
        _instrument_openai()
    elif library == "anthropic":
        _instrument_anthropic()
    elif library == "langchain":
        _instrument_langchain()
    elif library == "litellm":
        _instrument_litellm()
    else:
        print(f"[corerun Tracing] Unknown library: {library}")
        return

    _instrumented.add(library)


def _instrument_openai() -> None:
    """Instrument OpenAI client."""
    try:
        import openai

        original_create = openai.chat.completions.create.__func__ if hasattr(openai.chat.completions.create, '__func__') else None

        if original_create is None:
            # Try the instance method
            from openai import OpenAI
            client = OpenAI.__dict__.get('chat', None)
            if client is None:
                return

        def traced_create(self, *args, **kwargs):
            if not is_enabled():
                return original_create(self, *args, **kwargs)

            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            with start_span("openai.chat.completions.create", span_type=SpanType.CHAT_MODEL) as span:
                span.set_inputs({
                    "messages": messages,
                    "model": model,
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                })
                span.set_attribute("llm.provider", "openai")
                span.set_attribute("llm.model", model)

                try:
                    response = original_create(self, *args, **kwargs)

                    # Extract response data
                    if hasattr(response, "choices") and response.choices:
                        choice = response.choices[0]
                        span.set_outputs({
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                            },
                            "finish_reason": choice.finish_reason,
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                                "total_tokens": response.usage.total_tokens if response.usage else 0,
                            } if response.usage else {},
                        })

                    return response
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise

        # Patch the method
        if original_create:
            openai.chat.completions.create = traced_create

        print("[corerun Tracing] Instrumented OpenAI")

    except ImportError:
        print("[corerun Tracing] OpenAI not installed, skipping instrumentation")
    except Exception as e:
        print(f"[corerun Tracing] Failed to instrument OpenAI: {e}")


def _instrument_anthropic() -> None:
    """Instrument Anthropic client."""
    try:
        import anthropic

        original_create = anthropic.Anthropic.messages.create

        def traced_create(self, *args, **kwargs):
            if not is_enabled():
                return original_create(self, *args, **kwargs)

            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            with start_span("anthropic.messages.create", span_type=SpanType.CHAT_MODEL) as span:
                span.set_inputs({
                    "messages": messages,
                    "model": model,
                    "system": kwargs.get("system"),
                    "max_tokens": kwargs.get("max_tokens"),
                })
                span.set_attribute("llm.provider", "anthropic")
                span.set_attribute("llm.model", model)

                try:
                    response = original_create(self, *args, **kwargs)

                    if hasattr(response, "content") and response.content:
                        span.set_outputs({
                            "message": {
                                "role": "assistant",
                                "content": response.content[0].text if response.content else "",
                            },
                            "stop_reason": response.stop_reason,
                            "usage": {
                                "input_tokens": response.usage.input_tokens if response.usage else 0,
                                "output_tokens": response.usage.output_tokens if response.usage else 0,
                            } if response.usage else {},
                        })

                    return response
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise

        anthropic.Anthropic.messages.create = traced_create
        print("[corerun Tracing] Instrumented Anthropic")

    except ImportError:
        print("[corerun Tracing] Anthropic not installed, skipping instrumentation")
    except Exception as e:
        print(f"[corerun Tracing] Failed to instrument Anthropic: {e}")


def _instrument_langchain() -> None:
    """Instrument LangChain."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.callbacks.manager import CallbackManager

        class CoreRunCallbackHandler(BaseCallbackHandler):
            """LangChain callback handler for corerun tracing."""

            def __init__(self):
                self._spans: Dict[str, Span] = {}

            def on_llm_start(self, serialized, prompts, **kwargs):
                run_id = str(kwargs.get("run_id", uuid.uuid4()))
                span = Span(
                    span_id=run_id,
                    name="langchain.llm",
                    span_type=SpanType.LLM,
                    start_time=int(time.time() * 1000),
                    inputs={"prompts": prompts},
                )
                self._spans[run_id] = span
                _span_stack.append(span)

            def on_llm_end(self, response, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                if run_id in self._spans:
                    span = self._spans.pop(run_id)
                    span.set_outputs({"generations": str(response.generations)})
                    span.end(SpanStatus.OK)
                    if _span_stack and _span_stack[-1] == span:
                        _span_stack.pop()

            def on_llm_error(self, error, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                if run_id in self._spans:
                    span = self._spans.pop(run_id)
                    span.set_status(SpanStatus.ERROR, str(error))
                    span.end(SpanStatus.ERROR)
                    if _span_stack and _span_stack[-1] == span:
                        _span_stack.pop()

            def on_chain_start(self, serialized, inputs, **kwargs):
                run_id = str(kwargs.get("run_id", uuid.uuid4()))
                span = Span(
                    span_id=run_id,
                    name=serialized.get("name", "langchain.chain"),
                    span_type=SpanType.CHAIN,
                    start_time=int(time.time() * 1000),
                    inputs={"inputs": str(inputs)},
                )
                self._spans[run_id] = span

            def on_chain_end(self, outputs, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                if run_id in self._spans:
                    span = self._spans.pop(run_id)
                    span.set_outputs({"outputs": str(outputs)})
                    span.end(SpanStatus.OK)

            def on_tool_start(self, serialized, input_str, **kwargs):
                run_id = str(kwargs.get("run_id", uuid.uuid4()))
                span = Span(
                    span_id=run_id,
                    name=serialized.get("name", "langchain.tool"),
                    span_type=SpanType.TOOL,
                    start_time=int(time.time() * 1000),
                    inputs={"input": input_str},
                )
                self._spans[run_id] = span

            def on_tool_end(self, output, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                if run_id in self._spans:
                    span = self._spans.pop(run_id)
                    span.set_outputs({"output": str(output)})
                    span.end(SpanStatus.OK)

            def on_retriever_start(self, serialized, query, **kwargs):
                run_id = str(kwargs.get("run_id", uuid.uuid4()))
                span = Span(
                    span_id=run_id,
                    name="langchain.retriever",
                    span_type=SpanType.RETRIEVER,
                    start_time=int(time.time() * 1000),
                    inputs={"query": query},
                )
                self._spans[run_id] = span

            def on_retriever_end(self, documents, **kwargs):
                run_id = str(kwargs.get("run_id", ""))
                if run_id in self._spans:
                    span = self._spans.pop(run_id)
                    span.set_outputs({"documents": [str(d) for d in documents[:10]]})
                    span.end(SpanStatus.OK)

        # Store the handler globally so it can be used
        global _langchain_handler
        _langchain_handler = CoreRunCallbackHandler()

        print("[corerun Tracing] Instrumented LangChain (use get_langchain_handler() to get callback)")

    except ImportError:
        print("[corerun Tracing] LangChain not installed, skipping instrumentation")
    except Exception as e:
        print(f"[corerun Tracing] Failed to instrument LangChain: {e}")


def _instrument_litellm() -> None:
    """Instrument LiteLLM."""
    try:
        import litellm

        original_completion = litellm.completion

        def traced_completion(*args, **kwargs):
            if not is_enabled():
                return original_completion(*args, **kwargs)

            model = kwargs.get("model", args[0] if args else "unknown")
            messages = kwargs.get("messages", [])

            with start_span("litellm.completion", span_type=SpanType.CHAT_MODEL) as span:
                span.set_inputs({
                    "messages": messages,
                    "model": model,
                })
                span.set_attribute("llm.provider", "litellm")
                span.set_attribute("llm.model", model)

                try:
                    response = original_completion(*args, **kwargs)

                    if hasattr(response, "choices") and response.choices:
                        choice = response.choices[0]
                        span.set_outputs({
                            "message": {
                                "role": getattr(choice.message, "role", "assistant"),
                                "content": getattr(choice.message, "content", ""),
                            },
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                                "total_tokens": response.usage.total_tokens if response.usage else 0,
                            } if response.usage else {},
                        })

                    return response
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise

        litellm.completion = traced_completion
        print("[corerun Tracing] Instrumented LiteLLM")

    except ImportError:
        print("[corerun Tracing] LiteLLM not installed, skipping instrumentation")
    except Exception as e:
        print(f"[corerun Tracing] Failed to instrument LiteLLM: {e}")


def get_langchain_handler():
    """Get the LangChain callback handler for corerun tracing.

    Usage:
        from corerun.tracing import instrument, get_langchain_handler
        instrument("langchain")

        handler = get_langchain_handler()
        llm = ChatOpenAI(callbacks=[handler])
    """
    global _langchain_handler
    if "_langchain_handler" not in globals():
        _instrument_langchain()
    return _langchain_handler


# ========================
# Prompts Integration
# ========================

def load_prompt(
    prompt_id: str,
    version: Optional[int] = None,
    alias: Optional[str] = None,
    variables: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Load a prompt from the corerun Prompt Registry.

    Args:
        prompt_id: The prompt ID to load
        version: Specific version number (optional)
        alias: Alias to load (e.g., "production", "staging")
        variables: Variable values to substitute

    Returns:
        Dict containing prompt_type, template/messages, and rendered content

    Usage:
        prompt = load_prompt("summarization-prompt", alias="production")
        messages = prompt["rendered_messages"]
    """
    if not _tracing_client:
        raise RuntimeError("Tracing not configured. Call configure() first.")

    params = []
    if version:
        params.append(f"version={version}")
    elif alias:
        params.append(f"alias={alias}")

    query = f"?{'&'.join(params)}" if params else ""

    with httpx.Client(timeout=10.0) as client:
        response = client.get(
            f"{_tracing_client.api_url}/prompts/{prompt_id}/load{query}",
            headers={
                "Authorization": f"Bearer {_tracing_client.auth_token}",
                "X-Workspace-ID": _tracing_client.workspace_id,
            },
        )
        response.raise_for_status()
        prompt_data = response.json()

    # Render variables if provided
    if variables:
        if prompt_data.get("template"):
            template = prompt_data["template"]
            for key, value in variables.items():
                template = template.replace(f"{{{{{key}}}}}", value)
            prompt_data["rendered_template"] = template

        if prompt_data.get("messages"):
            rendered_messages = []
            for msg in prompt_data["messages"]:
                content = msg["content"]
                for key, value in variables.items():
                    content = content.replace(f"{{{{{key}}}}}", value)
                rendered_messages.append({"role": msg["role"], "content": content})
            prompt_data["rendered_messages"] = rendered_messages

    return prompt_data


# ========================
# Cleanup
# ========================

import atexit

def _cleanup():
    """Cleanup tracing on exit."""
    global _tracing_client
    if _tracing_client:
        _tracing_client.close()

atexit.register(_cleanup)
