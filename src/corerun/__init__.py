"""
corerun Python SDK

A Python client library for the corerun ML Platform.

Usage:
    import corerun

    # Initialize with API key
    corerun.init(auth_token="...", workspace="my-workspace")

    # Or use environment variables
    # CORERUN_AUTH_TOKEN, CORERUN_WORKSPACE

    # List datasets
    datasets = corerun.datasets.list()

    # Import from HuggingFace
    ds = corerun.datasets.from_huggingface("ylecun/mnist", name="mnist")

    # Submit a training job with experiment tracking
    job = corerun.jobs.submit(
        name="train-model",
        image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
        command=["python", "train.py"],
        gpu=1,
        experiment="my-experiment",  # Group outputs by experiment
        datasets=["mnist"],
    )

    # Wait for completion
    job.wait()

Inside training scripts:
    import corerun.tracking as track

    # Log metrics
    track.log_metric("loss", 0.5, step=epoch)

    # Save checkpoints
    track.save_checkpoint(model.state_dict(), "epoch_10.pt")

    # Get output directories
    output_dir = track.get_output_dir()

Model Registry:
    import corerun

    corerun.init()

    # Register a version of a model whose files are already in storage
    version = corerun.registry.create_version(
        "my-model",
        storage_path="models/my-model/1",
        framework="pytorch",
    )

    # Set alias and stage
    corerun.registry.set_alias("my-model", "champion", version.version)
    corerun.registry.set_version_stage("my-model", version.version, "production")

    # Moving files in and out is done from the CLI, which needs git-lfs:
    #   corerun models push my-model ./checkpoint
    #   corerun models pull my-model

Tracing (MLflow-compatible):
    import corerun
    from corerun import tracing

    # Initialize with tracing enabled
    corerun.init(auth_token="...", workspace="my-workspace")
    tracing.configure(enabled=True)

    # Decorator-based tracing
    @tracing.trace(span_type="LLM")
    def my_llm_call(messages):
        return response

    # Context manager tracing
    with tracing.start_span("embedding_step", span_type="EMBEDDING") as span:
        span.set_inputs({"text": "hello"})
        result = embed(text)
        span.set_outputs({"embedding": result})

    # Auto-instrumentation
    tracing.instrument("openai")  # Auto-trace OpenAI calls
    tracing.instrument("anthropic")  # Auto-trace Anthropic calls
    tracing.instrument("langchain")  # Auto-trace LangChain

Prompt Registry:
    import corerun
    from corerun import prompts

    corerun.init()

    # Load a prompt by alias
    prompt = prompts.load("summarization-prompt", alias="production")
    messages = prompt.render(document="...", focus_areas="key points")

    # List prompts
    all_prompts = prompts.list()

    # Create a prompt
    prompts.create(
        name="my-prompt",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "{{question}}"},
        ],
    )

MLflow Servers:
    import corerun

    corerun.init()

    # Create an MLflow server for your workspace
    server = corerun.mlflow.create(cluster_name="my-cluster")
    print(f"MLflow UI: {server.url}")

    # Get or create (returns existing if running)
    server = corerun.mlflow.get_or_create(cluster_name="my-cluster")

    # List servers
    servers = corerun.mlflow.list()

    # Get tracking URI for jobs
    uri = corerun.mlflow.get_tracking_uri()

    # Delete server
    corerun.mlflow.delete(server.id)

    Note: When an MLflow server exists, all jobs automatically receive
    MLFLOW_TRACKING_URI for seamless experiment tracking.

Notebooks:
    import corerun

    corerun.init()

    # Create a Jupyter notebook
    notebook = corerun.notebooks.create(
        name="data-exploration",
        compute_name="dgx-cluster",
        gpu=1,
        datasets=["mnist"],
    )

    # Wait for it to be ready
    notebook = corerun.notebooks.wait_for_running(notebook.id)
    print(f"Open: {notebook.url}")

    # Create VSCode notebook
    notebook = corerun.notebooks.create(
        name="dev-env",
        notebook_type="code-server",
        compute_name="dgx-cluster",
        gpu=1,
    )

    # Stop/start/delete
    corerun.notebooks.stop(notebook.id)
    corerun.notebooks.start(notebook.id)
    corerun.notebooks.delete(notebook.id)

Inference Servers:
    import corerun

    corerun.init()

    # Deploy a vLLM inference server
    server = corerun.inference.deploy(
        name="mistral-7b",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        compute_name="dgx-cluster",
        gpu=1,
        wait=True,
    )

    # Use with OpenAI client
    from openai import OpenAI
    client = OpenAI(base_url=server.external_path, api_key=server.api_key)
    response = client.chat.completions.create(
        model="mistral-7b",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Deploy from MLflow registry
    server = corerun.inference.deploy(
        name="finetuned-model",
        model_id="my-model@champion",
        model_source="mlflow",
        compute_name="dgx-cluster",
        gpu=1,
    )

    # Scale and manage
    corerun.inference.scale(server.id, min_replicas=2, max_replicas=4)
    corerun.inference.stop(server.id)
    corerun.inference.delete(server.id)
"""

__version__ = "0.1.0"

from corerun import (
    clusters,
    compute,
    datasets,
    endpoints,
    evaluations,
    finetune,
    inference,
    jobs,
    notebooks,
    prompts,
    quota,
    registry,
    tracing,
    tracking,
)
from corerun import mlflow_servers as mlflow
from corerun.client import CoreRunClient
from corerun.config import get_client, get_config, init

__all__ = [
    "__version__",
    "CoreRunClient",
    "init",
    "get_client",
    "get_config",
    "datasets",
    "jobs",
    "tracking",
    "registry",
    "tracing",
    "prompts",
    "mlflow",
    "notebooks",
    "endpoints",
    "inference",
    "finetune",
    "evaluations",
    "clusters",
    "compute",
    "quota",
]
