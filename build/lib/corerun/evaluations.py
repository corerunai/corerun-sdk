"""
Evaluations module for corerun SDK.

Provides functions for managing evaluation datasets and running LLM evaluations.

Usage:
    import corerun

    corerun.init(auth_token="...")

    # Create an evaluation dataset
    ds = corerun.evaluations.create_dataset(
        name="qa-bench",
        schema=[
            {"name": "input", "type": "string", "required": True},
            {"name": "expected_output", "type": "string", "required": True},
        ],
        examples=[
            {"input": "What is 2+2?", "expected_output": "4"},
        ],
    )

    # Create and start an evaluation run
    run = corerun.evaluations.create_run(
        name="mistral-eval",
        dataset_id=ds.id,
        model_config={
            "type": "inference_endpoint",
            "endpoint_id": "<server-id>",
            "model": "mistral-7b",
        },
        scorers=["correctness", "fluency"],
    )
    run = corerun.evaluations.start_run(run.id, compute_name="dgx-cluster")
    run = corerun.evaluations.wait_run(run.id)
    print(run.results)

    # List available scorers
    scorers = corerun.evaluations.list_scorers()
"""

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from corerun.config import get_client


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EvalDatasetColumn(BaseModel):
    """Schema column definition for an evaluation dataset."""
    name: str
    type: str                        # "string", "number", "json", "array"
    required: bool = False
    description: Optional[str] = None


class EvalDataset(BaseModel):
    """Evaluation dataset."""

    # The API calls this field "schema", which on a pydantic model shadows
    # BaseModel.schema() -- so pydantic warns about it, and the warning was
    # printed by every CLI command that imported this module. The attribute is
    # named for what it holds and aliased back to the wire name, so the API
    # contract is unchanged and callers may still pass schema=.
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    description: Optional[str] = None
    columns: List[EvalDatasetColumn] = Field(default_factory=list, alias="schema")
    example_count: int = 0
    tags: Optional[Dict[str, str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __repr__(self) -> str:
        return f"EvalDataset(name='{self.name}', examples={self.example_count})"


class EvalModelConfig(BaseModel):
    """Model configuration for an evaluation run."""
    type: str                           # "inference_endpoint" or "external_api"
    endpoint_id: Optional[str] = None  # corerun inference server ID
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str
    parameters: Optional[Dict[str, Any]] = None


class EvalRun(BaseModel):
    """Evaluation run."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    description: Optional[str] = None
    dataset_id: str = ""
    dataset_name: Optional[str] = None
    # A public benchmark the run measures, when it measures one instead of a
    # dataset the workspace made.
    benchmark: Optional[str] = None
    benchmark_limit: Optional[int] = None
    llm_config: EvalModelConfig = Field(alias="model_config")
    scorers: List[str] = []
    judge_config: Optional[Dict[str, Any]] = None
    status: str                         # "pending", "running", "completed", "failed"
    progress: int = 0
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @field_validator("scorers", mode="before")
    @classmethod
    def _no_scorers_is_empty(cls, value):
        """A benchmark run has no scorers of its own, and the API says so with
        null rather than an empty list."""
        return value or []

    @property
    def is_finished(self) -> bool:
        return self.status in ("completed", "failed")

    def __repr__(self) -> str:
        return f"EvalRun(name='{self.name}', status='{self.status}', progress={self.progress}%)"


class Scorer(BaseModel):
    """Available evaluation scorer."""
    id: str
    name: str
    description: str
    type: str   # "llm_judge", "exact", "metric"


# ---------------------------------------------------------------------------
# Dataset functions
# ---------------------------------------------------------------------------

def create_dataset(
    name: str,
    schema: List[Dict[str, Any]],
    description: Optional[str] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    tags: Optional[Dict[str, str]] = None,
    workspace: Optional[str] = None,
) -> EvalDataset:
    """
    Create an evaluation dataset.

    Args:
        name: Dataset name
        schema: Column definitions — list of dicts with keys:
            name (str), type (str: string/number/json/array),
            required (bool, optional), description (str, optional)
        description: Optional description
        examples: Initial example rows
        tags: Optional key-value tags
        workspace: Workspace ID (uses default if not specified)

    Returns:
        EvalDataset object

    Example:
        ds = corerun.evaluations.create_dataset(
            name="qa-benchmark",
            schema=[
                {"name": "input", "type": "string", "required": True},
                {"name": "expected_output", "type": "string", "required": True},
                {"name": "context", "type": "string"},
            ],
            examples=[
                {"input": "What is 2+2?", "expected_output": "4"},
                {"input": "Capital of France?", "expected_output": "Paris"},
            ],
        )
    """
    client = get_client()

    body: Dict[str, Any] = {
        "name": name,
        "schema": schema,
    }
    if description:
        body["description"] = description
    if examples:
        body["examples"] = examples
    if tags:
        body["tags"] = tags

    response = client.post("/evaluations/datasets", json=body, workspace=workspace)
    return EvalDataset(**response)


def list_datasets(workspace: Optional[str] = None) -> List[EvalDataset]:
    """
    List evaluation datasets in the workspace.

    Returns:
        List of EvalDataset objects

    Example:
        datasets = corerun.evaluations.list_datasets()
    """
    client = get_client()
    response = client.get("/evaluations/datasets", workspace=workspace)
    return [EvalDataset(**d) for d in response.get("datasets", [])]


def get_dataset(dataset_id: str, workspace: Optional[str] = None) -> EvalDataset:
    """
    Get an evaluation dataset by ID.

    Returns:
        EvalDataset object

    Example:
        ds = corerun.evaluations.get_dataset("abc123")
    """
    client = get_client()
    response = client.get(f"/evaluations/datasets/{dataset_id}", workspace=workspace)
    return EvalDataset(**response)


def delete_dataset(dataset_id: str, workspace: Optional[str] = None) -> None:
    """
    Delete an evaluation dataset.

    Example:
        corerun.evaluations.delete_dataset("abc123")
    """
    client = get_client()
    client.delete(f"/evaluations/datasets/{dataset_id}", workspace=workspace)


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def create_run(
    name: str,
    dataset_id: str = "",
    model_config: Dict[str, Any] = None,
    scorers: List[str] = None,
    description: Optional[str] = None,
    judge_config: Optional[Dict[str, Any]] = None,
    benchmark: str = "",
    benchmark_limit: int = 0,
    workspace: Optional[str] = None,
) -> EvalRun:
    """
    Create an evaluation run (starts in pending state).

    A run measures one of two things: a dataset the workspace made, scored by
    the scorers named beside it, or a public benchmark, which brings both its
    examples and its scorer. Say which with `dataset_id` or `benchmark`.

    Args:
        name: Run name
        dataset_id: Evaluation dataset ID (with `scorers`, for a dataset run)
        model_config: Model configuration dict with keys:
            type (str: "inference_endpoint" or "external_api"),
            model (str: model name),
            endpoint_id (str, for inference_endpoint),
            base_url (str, for external_api),
            api_key (str, optional),
            parameters (dict, optional)
        scorers: List of scorer IDs (e.g. ["correctness", "fluency"])
        description: Optional description
        judge_config: Optional judge model config dict
        benchmark: A public benchmark's name (e.g. "gsm8k"), instead of a dataset
        benchmark_limit: How many of its examples to run; 0 means all of them
        workspace: Workspace ID (uses default if not specified)

    Returns:
        EvalRun object (status=pending)

    Example:
        run = corerun.evaluations.create_run(
            name="my-eval",
            dataset_id="abc123",
            model_config={
                "type": "inference_endpoint",
                "endpoint_id": "<server-id>",
                "model": "mistral-7b",
            },
            scorers=["correctness", "fluency", "latency"],
        )

        # Or a public benchmark, which needs no dataset and no scorers:
        run = corerun.evaluations.create_run(
            name="gsm8k-check",
            benchmark="gsm8k",
            benchmark_limit=50,
            model_config={"type": "inference_endpoint", "endpoint_id": "...", "model": "..."},
        )
    """
    if not benchmark and not dataset_id:
        raise ValueError("a run needs a dataset_id or a benchmark")
    if dataset_id and not scorers:
        raise ValueError("a dataset run needs at least one scorer")

    client = get_client()

    body: Dict[str, Any] = {
        "name": name,
        "model_config": model_config or {},
    }
    if dataset_id:
        body["dataset_id"] = dataset_id
        body["scorers"] = scorers or []
    if benchmark:
        body["benchmark"] = benchmark
        if benchmark_limit:
            body["benchmark_limit"] = benchmark_limit
    if description:
        body["description"] = description
    if judge_config:
        body["judge_config"] = judge_config

    response = client.post("/evaluations/runs", json=body, workspace=workspace)
    return EvalRun(**response)


def list_runs(
    status: Optional[str] = None,
    workspace: Optional[str] = None,
) -> List[EvalRun]:
    """
    List evaluation runs in the workspace.

    Args:
        status: Filter by status (pending, running, completed, failed)
        workspace: Workspace ID (uses default if not specified)

    Returns:
        List of EvalRun objects

    Example:
        runs = corerun.evaluations.list_runs()
        completed = corerun.evaluations.list_runs(status="completed")
    """
    client = get_client()

    params: Dict[str, str] = {}
    if status:
        params["status"] = status

    response = client.get("/evaluations/runs", params=params, workspace=workspace)
    return [EvalRun(**r) for r in response.get("runs", [])]


def get_run(run_id: str, workspace: Optional[str] = None) -> EvalRun:
    """
    Get an evaluation run by ID.

    Returns:
        EvalRun object

    Example:
        run = corerun.evaluations.get_run("abc123")
        print(run.results)
    """
    client = get_client()
    response = client.get(f"/evaluations/runs/{run_id}", workspace=workspace)
    return EvalRun(**response)


def delete_run(run_id: str, workspace: Optional[str] = None) -> None:
    """
    Delete an evaluation run.

    Example:
        corerun.evaluations.delete_run("abc123")
    """
    client = get_client()
    client.delete(f"/evaluations/runs/{run_id}", workspace=workspace)


def start_run(
    run_id: str,
    compute_name: str,
    use_internal_network: bool = False,
    workspace: Optional[str] = None,
) -> EvalRun:
    """
    Start a pending evaluation run on a compute cluster.

    Args:
        run_id: Evaluation run ID
        compute_name: Compute target name (cluster)
        use_internal_network: Use cluster-internal URLs for inference (faster, no proxy)
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Updated EvalRun object (status=running)

    Example:
        run = corerun.evaluations.start_run("abc123", compute_name="dgx-cluster")
    """
    client = get_client()

    body: Dict[str, Any] = {
        "compute_name": compute_name,
        "use_internal_network": use_internal_network,
    }
    response = client.post(f"/evaluations/runs/{run_id}/start", json=body, workspace=workspace)
    return EvalRun(**response)


def wait_run(
    run_id: str,
    timeout: Optional[int] = None,
    poll_interval: int = 10,
    callback: Optional[Callable[[EvalRun], None]] = None,
    workspace: Optional[str] = None,
) -> EvalRun:
    """
    Wait for an evaluation run to complete.

    Args:
        run_id: Evaluation run ID
        timeout: Maximum wait time in seconds (None = infinite)
        poll_interval: Seconds between status checks (default: 10)
        callback: Function called on each poll with current EvalRun
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Final EvalRun object with results

    Raises:
        TimeoutError: If timeout exceeded

    Example:
        run = corerun.evaluations.wait_run("abc123", timeout=1800)
        if run.status == "completed":
            print(run.results["aggregate_scores"])
    """
    from corerun.exceptions import TimeoutError

    start = time.time()

    while True:
        run = get_run(run_id, workspace=workspace)

        if callback:
            callback(run)

        if run.is_finished:
            return run

        if timeout and (time.time() - start) > timeout:
            raise TimeoutError(f"Evaluation run {run_id} did not complete within {timeout}s")

        time.sleep(poll_interval)


def evaluate(
    name: str,
    dataset_id: str,
    model_config: Dict[str, Any],
    scorers: List[str],
    compute_name: str,
    description: Optional[str] = None,
    judge_config: Optional[Dict[str, Any]] = None,
    use_internal_network: bool = False,
    timeout: Optional[int] = None,
    workspace: Optional[str] = None,
) -> EvalRun:
    """
    Create, start, and wait for an evaluation run.

    Convenience function combining create_run(), start_run(), and wait_run().

    Returns:
        Completed EvalRun object with results

    Example:
        run = corerun.evaluations.evaluate(
            name="mistral-eval",
            dataset_id="abc123",
            model_config={
                "type": "inference_endpoint",
                "endpoint_id": "<server-id>",
                "model": "mistral-7b",
            },
            scorers=["correctness", "fluency"],
            compute_name="dgx-cluster",
        )
        print(run.results["aggregate_scores"])
    """
    run = create_run(
        name=name,
        dataset_id=dataset_id,
        model_config=model_config,
        scorers=scorers,
        description=description,
        judge_config=judge_config,
        workspace=workspace,
    )
    run = start_run(
        run.id,
        compute_name=compute_name,
        use_internal_network=use_internal_network,
        workspace=workspace,
    )
    return wait_run(run.id, timeout=timeout, workspace=workspace)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def list_scorers(workspace: Optional[str] = None) -> List[Scorer]:
    """
    List available evaluation scorers.

    Returns:
        List of Scorer objects

    Example:
        scorers = corerun.evaluations.list_scorers()
        for s in scorers:
            print(f"{s.id}: {s.description}")
    """
    client = get_client()
    response = client.get("/evaluations/scorers", workspace=workspace)
    return [Scorer(**s) for s in response.get("scorers", [])]
