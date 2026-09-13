"""
Inference Servers module for corerun SDK.

Provides functions for deploying and managing inference servers (vLLM, Ollama).

Usage:
    import corerun

    corerun.init(auth_token="...")

    # Deploy a vLLM inference server
    server = corerun.inference.deploy(
        name="mistral-7b",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        compute_name="dgx-cluster",
        gpu=1,
    )

    # Wait for it to be ready
    server = corerun.inference.wait_for_running(server.id)

    # Use the OpenAI-compatible API
    from openai import OpenAI
    client = OpenAI(base_url=server.external_url, api_key=server.api_key)
    response = client.chat.completions.create(
        model=server.model_id,
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Scale the server
    corerun.inference.scale(server.id, min_replicas=2, max_replicas=4)

    # Delete the server
    corerun.inference.delete(server.id)
"""

import time
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from corerun.config import get_client


class LoRAModuleInfo(BaseModel):
    """LoRA adapter information in server response."""
    name: str
    source: str
    path: Optional[str] = None
    job_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[int] = None


class InferenceServer(BaseModel):
    """Inference server information."""

    id: str
    name: str
    server_type: str  # vllm, ollama
    model_source: str  # huggingface, mlflow, registry, path
    model_id: str
    model_version: Optional[str] = None
    compute_name: str
    cluster_id: Optional[str] = None
    image: str
    gpu: float = 0
    replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 1
    status: str  # pending, running, stopped, failed
    internal_url: Optional[str] = None
    external_path: Optional[str] = None
    api_key: Optional[str] = None  # Only shown on create
    error: Optional[str] = None
    enable_tracing: bool = False
    lora_modules: List[LoRAModuleInfo] = []  # LoRA adapters loaded on base model
    owner_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.status == "running"

    @property
    def is_stopped(self) -> bool:
        """Check if server is stopped."""
        return self.status == "stopped"

    @property
    def is_failed(self) -> bool:
        """Check if server failed."""
        return self.status == "failed"

    @property
    def is_ready(self) -> bool:
        """Check if server is ready to receive requests."""
        return self.status == "running" and self.external_path is not None

    @property
    def openai_base_url(self) -> Optional[str]:
        """Get the OpenAI-compatible API base URL."""
        if self.external_path:
            # The external path typically includes /v1 for OpenAI compatibility
            return self.external_path.rstrip("/")
        return None

    def __repr__(self) -> str:
        return f"InferenceServer(name='{self.name}', model='{self.model_id}', status='{self.status}')"


class LoRAModule(BaseModel):
    """LoRA adapter configuration for fine-tuned models."""
    name: str  # Unique name for the adapter (used in API model field)
    source: str  # "job", "registry", "huggingface", "path"
    path: Optional[str] = None  # Direct path or HuggingFace ID
    job_id: Optional[str] = None  # Fine-tune job ID (when source="job")
    model_name: Optional[str] = None  # Registry model name (when source="registry")
    model_version: Optional[int] = None  # Registry version (when source="registry")


class CreateInferenceServerRequest(BaseModel):
    """Request to create an inference server."""
    name: str
    server_type: str = "vllm"  # vllm, ollama
    model_source: str = "huggingface"  # huggingface, mlflow, registry, path
    model_id: str
    model_version: Optional[str] = None
    compute_name: str
    cluster_id: Optional[str] = None
    image: Optional[str] = None
    gpu: float = 1
    min_replicas: int = 1
    max_replicas: int = 1
    endpoint_name: Optional[str] = None
    extra_args: Optional[List[str]] = None
    served_model_names: Optional[List[str]] = None

    # vLLM-specific options
    max_model_len: Optional[int] = None
    tensor_parallel: Optional[int] = None
    quantization: Optional[str] = None
    gpu_memory_util: Optional[float] = None
    enforce_eager: bool = False  # Disable CUDA graphs for unsupported GPUs

    # LoRA adapter configuration
    lora_modules: Optional[List[LoRAModule]] = None

    # MLflow tracing
    enable_tracing: bool = False


class ScaleRequest(BaseModel):
    """Request to scale an inference server."""
    min_replicas: int
    max_replicas: int


def _server_from_response(data: dict) -> InferenceServer:
    """Convert API response to InferenceServer object."""
    # Parse LoRA modules
    lora_modules = []
    if "lora_modules" in data and data["lora_modules"]:
        lora_modules = [
            LoRAModuleInfo(
                name=m.get("name", ""),
                source=m.get("source", ""),
                path=m.get("path"),
                job_id=m.get("job_id"),
                model_name=m.get("model_name"),
                model_version=m.get("model_version"),
            )
            for m in data["lora_modules"]
        ]

    return InferenceServer(
        id=data.get("id", ""),
        name=data.get("name", ""),
        server_type=data.get("server_type", "vllm"),
        model_source=data.get("model_source", "huggingface"),
        model_id=data.get("model_id", ""),
        model_version=data.get("model_version"),
        compute_name=data.get("compute_name", ""),
        cluster_id=data.get("cluster_id"),
        image=data.get("image", ""),
        gpu=data.get("gpu", 0),
        replicas=data.get("replicas", 1),
        min_replicas=data.get("min_replicas", 1),
        max_replicas=data.get("max_replicas", 1),
        status=data.get("status", "pending"),
        internal_url=data.get("internal_url"),
        external_path=data.get("external_path"),
        api_key=data.get("api_key"),
        error=data.get("error"),
        enable_tracing=data.get("enable_tracing", False),
        lora_modules=lora_modules,
        owner_id=data.get("owner_id", ""),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        started_at=data.get("started_at"),
    )


def list(
    status: Optional[str] = None,
    workspace: Optional[str] = None,
) -> List[InferenceServer]:
    """
    List inference servers in the workspace.

    Args:
        status: Filter by status (pending, running, stopped, failed)
        workspace: Workspace ID (uses default if not specified)

    Returns:
        List of InferenceServer objects

    Example:
        # All servers
        servers = corerun.inference.list()

        # Running servers only
        running = corerun.inference.list(status="running")
    """
    client = get_client()

    params = {}
    if status:
        params["status"] = status

    response = client.get("/inference-servers", params=params, workspace=workspace)
    return [_server_from_response(s) for s in response.get("servers", [])]


def get(server_id: str, workspace: Optional[str] = None) -> InferenceServer:
    """
    Get an inference server by ID.

    Args:
        server_id: Server ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        InferenceServer object

    Example:
        server = corerun.inference.get("abc123")
        print(f"Status: {server.status}")
    """
    client = get_client()
    response = client.get(f"/inference-servers/{server_id}", workspace=workspace)
    return _server_from_response(response)


def deploy(
    name: str,
    model_id: str,
    compute_name: str,
    server_type: str = "vllm",
    model_source: str = "huggingface",
    model_version: Optional[str] = None,
    image: Optional[str] = None,
    gpu: float = 1,
    min_replicas: int = 1,
    max_replicas: int = 1,
    max_model_len: Optional[int] = None,
    tensor_parallel: Optional[int] = None,
    quantization: Optional[str] = None,
    gpu_memory_util: Optional[float] = None,
    enforce_eager: bool = False,
    lora_modules: Optional[List[LoRAModule]] = None,
    enable_tracing: bool = False,
    endpoint: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    served_names: Optional[List[str]] = None,
    wait: bool = False,
    timeout: int = 600,
    workspace: Optional[str] = None,
) -> InferenceServer:
    """
    Deploy a new inference server.

    Args:
        name: Server name
        model_id: Model identifier (HuggingFace ID, MLflow URI, or registry name)
        compute_name: Compute target name (cluster)
        server_type: Server type ("vllm" or "ollama")
        model_source: Model source ("huggingface", "mlflow", "registry", "path").
            "path" serves weights already on the machine — model_id is then the
            directory, and nothing is downloaded.
        served_names: What callers ask for, when that differs from model_id.
            Required in spirit for path-sourced models, which would otherwise
            be published under their directory. Several are allowed and the
            endpoint answers to each.
        extra_args: Flags passed to the serving engine exactly as given, after
            everything the platform sets — so a repeated flag resolves to yours.
            The platform models a handful of vLLM's options; this is for the
            rest, e.g. ["--kv-cache-dtype", "fp8", "--enable-prefix-caching"].
        endpoint: The endpoint this model answers behind. Defaults to a new one
            named after the deployment; naming an existing one adds this model
            to it, and callers pick between them by model name.
        model_version: Version for registry models
        image: Docker image (uses default for server type if not specified)
        gpu: Number of GPUs
        min_replicas: Minimum number of replicas for autoscaling
        max_replicas: Maximum number of replicas for autoscaling
        max_model_len: Maximum model context length (vLLM)
        tensor_parallel: Tensor parallel size (vLLM)
        quantization: Quantization method (vLLM: "awq", "squeezellm", "gptq")
        gpu_memory_util: GPU memory utilization (vLLM, 0.0-1.0)
        enforce_eager: Disable CUDA graphs (vLLM, for unsupported GPUs)
        lora_modules: List of LoRA adapters to load on top of base model
        enable_tracing: Enable MLflow tracing
        wait: Wait for server to be running (default: False)
        timeout: Timeout in seconds when waiting (default: 600)
        workspace: Workspace ID (uses default if not specified)

    Returns:
        InferenceServer object (includes api_key only on create)

    Example:
        # Deploy a HuggingFace model
        server = corerun.inference.deploy(
            name="mistral-7b",
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            compute_name="dgx-cluster",
            gpu=1,
            wait=True,
        )

        # Deploy from MLflow registry
        server = corerun.inference.deploy(
            name="my-finetuned-model",
            model_id="my-model@champion",  # MLflow alias
            model_source="mlflow",
            compute_name="dgx-cluster",
            gpu=1,
        )

        # Deploy with autoscaling and tracing
        server = corerun.inference.deploy(
            name="production-llm",
            model_id="meta-llama/Llama-2-7b-chat-hf",
            compute_name="dgx-cluster",
            gpu=1,
            min_replicas=2,
            max_replicas=8,
            enable_tracing=True,
        )

        # Deploy with LoRA adapter from fine-tune job
        server = corerun.inference.deploy(
            name="finetuned-llm",
            model_id="meta-llama/Llama-2-7b-chat-hf",  # Base model
            compute_name="dgx-cluster",
            gpu=1,
            lora_modules=[
                LoRAModule(
                    name="my-adapter",
                    source="job",
                    job_id="abc123",  # Fine-tune job ID
                ),
            ],
        )

        # Use the server with OpenAI client
        from openai import OpenAI
        client = OpenAI(
            base_url=f"{server.external_path}",
            api_key=server.api_key,
        )
    """
    client = get_client()

    request = CreateInferenceServerRequest(
        name=name,
        server_type=server_type,
        model_source=model_source,
        model_id=model_id,
        model_version=model_version,
        compute_name=compute_name,
        image=image,
        gpu=gpu,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        max_model_len=max_model_len,
        tensor_parallel=tensor_parallel,
        quantization=quantization,
        gpu_memory_util=gpu_memory_util,
        enforce_eager=enforce_eager,
        lora_modules=lora_modules,
        enable_tracing=enable_tracing,
        endpoint_name=endpoint,
        extra_args=extra_args,
        served_model_names=served_names,
    )

    response = client.post(
        "/inference-servers",
        json=request.model_dump(exclude_none=True),
        workspace=workspace,
    )
    server = _server_from_response(response)

    if wait:
        server = wait_for_running(server.id, timeout=timeout, workspace=workspace)

    return server


def scale(
    server_id: str,
    min_replicas: int,
    max_replicas: int,
    workspace: Optional[str] = None,
) -> InferenceServer:
    """
    Scale an inference server.

    Args:
        server_id: Server ID
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Updated InferenceServer object

    Example:
        server = corerun.inference.scale(
            "abc123",
            min_replicas=2,
            max_replicas=4,
        )
    """
    client = get_client()

    request = ScaleRequest(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )

    response = client.post(
        f"/inference-servers/{server_id}/scale",
        json=request.model_dump(),
        workspace=workspace,
    )
    return _server_from_response(response)


def stop(server_id: str, workspace: Optional[str] = None) -> InferenceServer:
    """
    Stop an inference server.

    Args:
        server_id: Server ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Updated InferenceServer object

    Example:
        server = corerun.inference.stop("abc123")
    """
    client = get_client()
    response = client.post(f"/inference-servers/{server_id}/stop", workspace=workspace)
    return _server_from_response(response)


def start(server_id: str, workspace: Optional[str] = None) -> InferenceServer:
    """
    Start a stopped inference server.

    Args:
        server_id: Server ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Updated InferenceServer object

    Example:
        server = corerun.inference.start("abc123")
    """
    client = get_client()
    response = client.post(f"/inference-servers/{server_id}/start", workspace=workspace)
    return _server_from_response(response)


def restart(server_id: str, workspace: Optional[str] = None) -> dict:
    """
    Perform a rolling restart of a running inference server.

    Args:
        server_id: Server ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Dict with message and server_id

    Example:
        corerun.inference.restart("abc123")
    """
    client = get_client()
    return client.post(f"/inference-servers/{server_id}/restart", workspace=workspace)


def list_types(workspace: Optional[str] = None) -> list:
    """
    List available inference server types and their default images.

    Returns:
        List of server type dicts (server_type, image, description)

    Example:
        types = corerun.inference.list_types()
        for t in types:
            print(t['server_type'], t['image'])
    """
    client = get_client()
    response = client.get("/inference-servers/types", workspace=workspace)
    types = response.get("types", response) if isinstance(response, dict) else response
    # Normalize field names (API uses "name"/"default_image", SDK exposes "server_type"/"image")
    return [
        {
            "server_type": t.get("name", t.get("server_type", "")),
            "display_name": t.get("display_name", ""),
            "description": t.get("description", ""),
            "image": t.get("default_image", t.get("image", "")),
            "api_format": t.get("api_format", ""),
            "features": t.get("features", []),
        }
        for t in (types or [])
    ]


def delete(server_id: str, workspace: Optional[str] = None) -> None:
    """
    Delete an inference server.

    Args:
        server_id: Server ID
        workspace: Workspace ID (uses default if not specified)

    Example:
        corerun.inference.delete("abc123")
    """
    client = get_client()
    client.delete(f"/inference-servers/{server_id}", workspace=workspace)


def wait_for_running(
    server_id: str,
    timeout: int = 600,
    poll_interval: int = 10,
    workspace: Optional[str] = None,
) -> InferenceServer:
    """
    Wait for an inference server to be running.

    Args:
        server_id: Server ID
        timeout: Maximum wait time in seconds (default: 600, models take time to load)
        poll_interval: Seconds between status checks
        workspace: Workspace ID (uses default if not specified)

    Returns:
        InferenceServer object (in running state)

    Raises:
        TimeoutError: If server doesn't start within timeout
        Exception: If server fails to start

    Example:
        server = corerun.inference.deploy(...)
        server = corerun.inference.wait_for_running(server.id)
        print(f"API endpoint: {server.external_path}")
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        server = get(server_id, workspace=workspace)

        if server.is_running and server.external_path:
            return server

        if server.is_failed:
            raise Exception(f"Inference server failed: {server.error}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Inference server {server_id} did not start within {timeout}s")


def regenerate_api_key(
    server_id: str,
    workspace: Optional[str] = None,
) -> str:
    """
    Regenerate the API key for an inference server.

    Args:
        server_id: Server ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        New API key string

    Example:
        new_key = corerun.inference.regenerate_api_key("abc123")
        print(f"New API key: {new_key}")
    """
    client = get_client()
    response = client.post(
        f"/inference-servers/{server_id}/regenerate-key",
        workspace=workspace,
    )
    return response.get("api_key", "")
