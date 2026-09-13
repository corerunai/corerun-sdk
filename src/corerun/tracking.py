"""
Experiment tracking utilities for use inside corerun jobs.

This module provides helper functions for tracking experiments, logging metrics,
and saving artifacts from within training code running on corerun.

Supports multiple tracking backends:
- native: File-based tracking (default) - saves to /logs, /outputs, /checkpoints, /artifacts
- mlflow: MLflow tracking server integration
- both: Log to both native files and MLflow

Usage inside a training script:
    import corerun.tracking as track

    # Configure backend (optional - defaults to "native")
    # Auto-configures MLflow if MLFLOW_TRACKING_URI is set
    track.configure(backend="both")  # or "native" or "mlflow"

    # Log metrics
    track.log_metric("loss", 0.5)
    track.log_metric("accuracy", 0.95)

    # Log metrics with step
    for epoch in range(100):
        track.log_metric("train_loss", loss, step=epoch)

    # Save a checkpoint
    track.save_checkpoint(model.state_dict(), "epoch_10.pt")

    # Save model output
    track.save_output("predictions.csv", predictions_df.to_csv())

    # Save artifact
    track.save_artifact("confusion_matrix.png", fig_bytes)

    # Log parameters
    track.log_params({"lr": 0.001, "batch_size": 32})
"""

import os
import json
import atexit
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


# Tracking backend configuration
_backend: str = "native"  # "native", "mlflow", "both"
_mlflow_enabled: bool = False
_mlflow_run_started: bool = False
_auto_configured: bool = False


def configure(
    backend: str = "native",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    auto_start_run: bool = True,
) -> None:
    """
    Configure experiment tracking backend.

    Args:
        backend: Tracking backend - "native" (file-based), "mlflow", or "both"
        mlflow_tracking_uri: MLflow server URL (defaults to MLFLOW_TRACKING_URI env var)
        mlflow_experiment_name: MLflow experiment name (defaults to CORERUN_EXPERIMENT_NAME)
        auto_start_run: Automatically start MLflow run on first log call

    Example:
        # Use native file-based tracking (default)
        track.configure(backend="native")

        # Use MLflow only
        track.configure(backend="mlflow", mlflow_tracking_uri="http://mlflow:5000")

        # Use both native and MLflow
        track.configure(backend="both")
    """
    global _backend, _mlflow_enabled, _auto_configured

    _backend = backend
    _auto_configured = True

    if backend in ("mlflow", "both"):
        _setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name, auto_start_run)


def _setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    auto_start_run: bool = True,
) -> bool:
    """Setup MLflow client configuration."""
    global _mlflow_enabled

    try:
        import mlflow
    except ImportError:
        print("Warning: mlflow not installed. Install with: "
              'pip install "corerun[mlflow] @ git+https://github.com/corerunai/corerun-sdk.git"')
        _mlflow_enabled = False
        return False

    # Set tracking URI
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)

    # Set experiment name
    exp_name = experiment_name or os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        get_experiment_name()
    )
    if exp_name:
        mlflow.set_experiment(exp_name)

    _mlflow_enabled = True

    # Register cleanup on exit
    atexit.register(_end_mlflow_run)

    return True


def _auto_configure() -> None:
    """Auto-configure backend based on environment variables."""
    global _auto_configured, _backend

    if _auto_configured:
        return

    _auto_configured = True

    # Auto-enable MLflow if tracking URI is set
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        # Default to "both" when MLflow is available
        _backend = "both"
        _setup_mlflow(mlflow_uri)


def _ensure_mlflow_run() -> None:
    """Start MLflow run if not already active."""
    global _mlflow_run_started

    if not _mlflow_enabled or _mlflow_run_started:
        return

    try:
        import mlflow

        if mlflow.active_run() is None:
            # Create run name from corerun context
            job_id = get_job_id()
            exp_name = get_experiment_name()
            run_name = f"{exp_name}-{job_id}" if job_id else exp_name

            mlflow.start_run(run_name=run_name)

            # Log corerun context as tags
            tags = {"corerun.experiment": exp_name}
            if job_id:
                tags["corerun.job_id"] = job_id
            workspace_id = get_workspace_id()
            if workspace_id:
                tags["corerun.workspace_id"] = workspace_id

            mlflow.set_tags(tags)

        _mlflow_run_started = True
    except Exception as e:
        print(f"Warning: Failed to start MLflow run: {e}")


def _end_mlflow_run() -> None:
    """End MLflow run if active."""
    global _mlflow_run_started

    if not _mlflow_enabled or not _mlflow_run_started:
        return

    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()
        _mlflow_run_started = False
    except Exception:
        pass


def get_backend() -> str:
    """Get the current tracking backend configuration."""
    return _backend


def is_mlflow_enabled() -> bool:
    """Check if MLflow tracking is enabled."""
    return _mlflow_enabled


def _get_output_dir() -> Path:
    """Get the output directory from environment or default."""
    return Path(os.environ.get("CORERUN_OUTPUT_DIR", "/outputs"))


def _get_checkpoint_dir() -> Path:
    """Get the checkpoint directory from environment or default."""
    return Path(os.environ.get("CORERUN_CHECKPOINT_DIR", "/checkpoints"))


def _get_log_dir() -> Path:
    """Get the log directory from environment or default."""
    return Path(os.environ.get("CORERUN_LOG_DIR", "/logs"))


def _get_artifact_dir() -> Path:
    """Get the artifact directory from environment or default."""
    return Path(os.environ.get("CORERUN_ARTIFACT_DIR", "/artifacts"))


def get_experiment_name() -> str:
    """Get the current experiment name from environment."""
    return os.environ.get("CORERUN_EXPERIMENT_NAME", "default")


def get_job_id() -> Optional[str]:
    """Get the current job ID from environment."""
    return os.environ.get("CORERUN_JOB_ID")


def get_workspace_id() -> Optional[str]:
    """Get the current workspace ID from environment."""
    return os.environ.get("CORERUN_WORKSPACE_ID")


def log_metric(
    name: str,
    value: float,
    step: Optional[int] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    """
    Log a metric value.

    Depending on the configured backend:
    - native: Saves to /logs/metrics.jsonl
    - mlflow: Logs to MLflow tracking server
    - both: Logs to both

    Args:
        name: Metric name (e.g., "loss", "accuracy")
        value: Metric value
        step: Optional step number (epoch, iteration)
        timestamp: Optional timestamp (defaults to now)

    Example:
        track.log_metric("train_loss", 0.5, step=10)
        track.log_metric("accuracy", 0.95)
    """
    _auto_configure()

    # Native tracking
    if _backend in ("native", "both"):
        _log_metric_native(name, value, step, timestamp)

    # MLflow tracking
    if _backend in ("mlflow", "both") and _mlflow_enabled:
        _log_metric_mlflow(name, value, step)


def _log_metric_native(
    name: str,
    value: float,
    step: Optional[int] = None,
    timestamp: Optional[datetime] = None,
) -> None:
    """Log metric to native file-based storage."""
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = log_dir / "metrics.jsonl"

    entry = {
        "name": name,
        "value": value,
        "timestamp": (timestamp or datetime.utcnow()).isoformat(),
    }
    if step is not None:
        entry["step"] = step

    with open(metrics_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _log_metric_mlflow(name: str, value: float, step: Optional[int] = None) -> None:
    """Log metric to MLflow."""
    _ensure_mlflow_run()
    try:
        import mlflow
        mlflow.log_metric(name, value, step=step)
    except Exception as e:
        print(f"Warning: Failed to log metric to MLflow: {e}")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log multiple metrics at once.

    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number

    Example:
        track.log_metrics({
            "train_loss": 0.5,
            "val_loss": 0.6,
            "accuracy": 0.95
        }, step=10)
    """
    _auto_configure()
    timestamp = datetime.utcnow()

    # Native tracking
    if _backend in ("native", "both"):
        for name, value in metrics.items():
            _log_metric_native(name, value, step=step, timestamp=timestamp)

    # MLflow tracking (batch)
    if _backend in ("mlflow", "both") and _mlflow_enabled:
        _ensure_mlflow_run()
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow: {e}")


def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters/hyperparameters.

    Depending on the configured backend:
    - native: Saves to /logs/params.json
    - mlflow: Logs to MLflow tracking server
    - both: Logs to both

    Args:
        params: Dictionary of parameter name -> value

    Example:
        track.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam"
        })
    """
    _auto_configure()

    # Native tracking
    if _backend in ("native", "both"):
        _log_params_native(params)

    # MLflow tracking
    if _backend in ("mlflow", "both") and _mlflow_enabled:
        _log_params_mlflow(params)


def _log_params_native(params: Dict[str, Any]) -> None:
    """Log parameters to native file-based storage."""
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    params_file = log_dir / "params.json"

    # Load existing params if any
    existing = {}
    if params_file.exists():
        with open(params_file) as f:
            existing = json.load(f)

    # Merge and save
    existing.update(params)
    with open(params_file, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def _log_params_mlflow(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    _ensure_mlflow_run()
    try:
        import mlflow
        # Convert all values to strings for MLflow
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)
    except Exception as e:
        print(f"Warning: Failed to log params to MLflow: {e}")


def save_checkpoint(
    data: Any,
    name: str,
    framework: str = "auto",
    log_to_mlflow: bool = True,
) -> Path:
    """
    Save a training checkpoint.

    Supports PyTorch, TensorFlow, and raw bytes.
    Optionally logs to MLflow if enabled.

    Args:
        data: Model state dict (PyTorch), model (TensorFlow), or bytes
        name: Checkpoint filename (e.g., "epoch_10.pt", "best_model.h5")
        framework: "pytorch", "tensorflow", or "auto" for auto-detection
        log_to_mlflow: Whether to also log to MLflow (default: True)

    Returns:
        Path to saved checkpoint

    Example:
        # PyTorch
        track.save_checkpoint(model.state_dict(), "epoch_10.pt")

        # Full PyTorch model + optimizer
        track.save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 10,
        }, "checkpoint_10.pt")

        # TensorFlow
        track.save_checkpoint(model, "model.h5")
    """
    _auto_configure()

    checkpoint_dir = _get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / name

    if framework == "auto":
        # Try to detect framework
        if hasattr(data, "save"):
            # TensorFlow/Keras model
            framework = "tensorflow"
        elif isinstance(data, dict) or hasattr(data, "state_dict"):
            framework = "pytorch"
        else:
            framework = "raw"

    if framework == "pytorch":
        import torch
        torch.save(data, path)
    elif framework == "tensorflow":
        data.save(str(path))
    elif isinstance(data, bytes):
        with open(path, "wb") as f:
            f.write(data)
    else:
        # Try pickle
        import pickle
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # Log to MLflow if enabled
    if log_to_mlflow and _backend in ("mlflow", "both") and _mlflow_enabled:
        _log_artifact_mlflow(path)

    return path


def save_output(
    name: str,
    content: Union[str, bytes],
    mode: str = "auto",
) -> Path:
    """
    Save training output (model, predictions, etc.).

    Args:
        name: Output filename
        content: Content to save (string or bytes)
        mode: "text", "binary", or "auto"

    Returns:
        Path to saved output

    Example:
        # Save predictions
        track.save_output("predictions.csv", df.to_csv())

        # Save model file
        track.save_output("model.onnx", onnx_bytes)
    """
    output_dir = _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / name

    if mode == "auto":
        mode = "binary" if isinstance(content, bytes) else "text"

    if mode == "binary":
        with open(path, "wb") as f:
            f.write(content)
    else:
        with open(path, "w") as f:
            f.write(content)

    return path


def save_artifact(
    name: str,
    content: Union[str, bytes],
    mode: str = "auto",
    log_to_mlflow: bool = True,
) -> Path:
    """
    Save an artifact (plots, figures, reports, etc.).

    Depending on the configured backend:
    - native: Saves to /artifacts/
    - mlflow: Logs artifact to MLflow tracking server
    - both: Saves locally and logs to MLflow

    Args:
        name: Artifact filename
        content: Content to save
        mode: "text", "binary", or "auto"
        log_to_mlflow: Whether to also log to MLflow (default: True)

    Returns:
        Path to saved artifact

    Example:
        # Save a matplotlib figure
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        track.save_artifact("training_curve.png", buf.getvalue())

        # Save a report
        track.save_artifact("report.html", html_content)
    """
    _auto_configure()

    # Always save to native storage first
    artifact_dir = _get_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    path = artifact_dir / name

    if mode == "auto":
        mode = "binary" if isinstance(content, bytes) else "text"

    if mode == "binary":
        with open(path, "wb") as f:
            f.write(content)
    else:
        with open(path, "w") as f:
            f.write(content)

    # Log to MLflow if enabled
    if log_to_mlflow and _backend in ("mlflow", "both") and _mlflow_enabled:
        _log_artifact_mlflow(path)

    return path


def _log_artifact_mlflow(path: Path) -> None:
    """Log an artifact file to MLflow."""
    _ensure_mlflow_run()
    try:
        import mlflow
        mlflow.log_artifact(str(path))
    except Exception as e:
        print(f"Warning: Failed to log artifact to MLflow: {e}")


def get_output_dir() -> Path:
    """
    Get the output directory path.

    Use this to save outputs directly using your own code.

    Returns:
        Path to output directory

    Example:
        output_dir = track.get_output_dir()
        torch.save(model.state_dict(), output_dir / "model.pt")
    """
    path = _get_output_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_dir() -> Path:
    """
    Get the checkpoint directory path.

    Use this to save checkpoints directly using your own code.

    Returns:
        Path to checkpoint directory

    Example:
        checkpoint_dir = track.get_checkpoint_dir()
        trainer.save_checkpoint(checkpoint_dir / "latest.ckpt")
    """
    path = _get_checkpoint_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_dir() -> Path:
    """
    Get the log directory path.

    Useful for TensorBoard, WandB, etc.

    Returns:
        Path to log directory

    Example:
        log_dir = track.get_log_dir()
        writer = SummaryWriter(log_dir=str(log_dir))
    """
    path = _get_log_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_artifact_dir() -> Path:
    """
    Get the artifact directory path.

    Returns:
        Path to artifact directory

    Example:
        artifact_dir = track.get_artifact_dir()
        plt.savefig(artifact_dir / "plot.png")
    """
    path = _get_artifact_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
