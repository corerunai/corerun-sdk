"""
MLflow Server Management for corerun SDK.

Provides functions to create, list, and manage MLflow tracking servers
for your workspace. MLflow servers provide:
- Experiment tracking (metrics, parameters, artifacts)
- LLM tracing (observability for AI agents and RAG)
- Model registry

Usage:
    import corerun

    corerun.init(auth_token="...")

    # Create an MLflow server
    server = corerun.mlflow.create(cluster_name="my-cluster")
    print(f"MLflow UI: {server.url}")

    # List servers
    servers = corerun.mlflow.list()

    # Delete a server
    corerun.mlflow.delete(server.id)

Note:
    When an MLflow server exists for your workspace, all jobs submitted
    to that workspace automatically receive MLFLOW_TRACKING_URI and
    MLFLOW_EXPERIMENT_NAME environment variables for seamless tracking.
"""

import time
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

from corerun.config import get_client


class MLflowServer(BaseModel):
    """MLflow tracking server."""

    id: str
    server_id: str  # Same as id, for compatibility
    workspace_id: str
    cluster_name: str
    status: str  # pending, running, stopped, failed
    url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.status == "running"

    @property
    def is_failed(self) -> bool:
        """Check if server failed."""
        return self.status == "failed"


def _server_from_response(data: dict) -> MLflowServer:
    """Convert API response to MLflowServer object."""
    return MLflowServer(
        id=data.get("server_id", data.get("id", "")),
        server_id=data.get("server_id", data.get("id", "")),
        workspace_id=data.get("workspace_id", ""),
        cluster_name=data.get("cluster_name", ""),
        status=data.get("status", "unknown"),
        url=data.get("url"),
        error=data.get("error"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def list(workspace: Optional[str] = None) -> List[MLflowServer]:
    """
    List MLflow servers in the workspace.

    Args:
        workspace: Workspace ID (uses default if not specified)

    Returns:
        List of MLflowServer objects

    Example:
        servers = corerun.mlflow.list()
        for s in servers:
            print(f"{s.id}: {s.status}")
    """
    client = get_client()
    response = client.get("/mlflow-servers", workspace=workspace)
    return [_server_from_response(s) for s in response.get("servers", [])]


def get(server_id: str, workspace: Optional[str] = None) -> MLflowServer:
    """
    Get an MLflow server by ID.

    Args:
        server_id: Server ID
        workspace: Workspace ID (uses default if not specified)

    Returns:
        MLflowServer object

    Example:
        server = corerun.mlflow.get("abc123")
        print(f"Status: {server.status}")
    """
    client = get_client()
    response = client.get(f"/mlflow-servers/{server_id}", workspace=workspace)
    return _server_from_response(response)


def create(
    cluster_name: str,
    workspace: Optional[str] = None,
    wait: bool = True,
    timeout: int = 120,
) -> MLflowServer:
    """
    Create an MLflow tracking server for the workspace.

    The server is deployed on the specified cluster and provides:
    - Experiment tracking (metrics, parameters, artifacts)
    - LLM tracing (observability for AI agents)
    - Model registry

    Args:
        cluster_name: Name of the cluster to deploy on
        workspace: Workspace ID (uses default if not specified)
        wait: Wait for server to be running (default: True)
        timeout: Timeout in seconds when waiting (default: 120)

    Returns:
        MLflowServer object

    Note:
        Only one MLflow server can be active per workspace. If one already
        exists, this will return the existing server.

    Example:
        # Create and wait for server
        server = corerun.mlflow.create("my-cluster")
        print(f"MLflow UI: {server.url}")

        # Create without waiting
        server = corerun.mlflow.create("my-cluster", wait=False)
    """
    client = get_client()

    # Check if server already exists
    existing = list(workspace=workspace)
    running = [s for s in existing if s.is_running]
    if running:
        return running[0]

    # Create new server
    response = client.post(
        "/mlflow-servers",
        json={"cluster_name": cluster_name},
        workspace=workspace,
    )
    server = _server_from_response(response)

    if wait:
        server = wait_for_running(server.id, timeout=timeout, workspace=workspace)

    return server


def delete(server_id: str, workspace: Optional[str] = None) -> None:
    """
    Delete an MLflow server.

    Args:
        server_id: Server ID to delete
        workspace: Workspace ID (uses default if not specified)

    Example:
        corerun.mlflow.delete("abc123")
    """
    client = get_client()
    client.delete(f"/mlflow-servers/{server_id}", workspace=workspace)


def wait_for_running(
    server_id: str,
    timeout: int = 120,
    poll_interval: int = 5,
    workspace: Optional[str] = None,
) -> MLflowServer:
    """
    Wait for an MLflow server to be running.

    Args:
        server_id: Server ID
        timeout: Maximum wait time in seconds
        poll_interval: Seconds between status checks
        workspace: Workspace ID (uses default if not specified)

    Returns:
        MLflowServer object (in running state)

    Raises:
        TimeoutError: If server doesn't start within timeout
        Exception: If server fails to start

    Example:
        server = corerun.mlflow.wait_for_running("abc123")
    """
    start = time.time()

    while time.time() - start < timeout:
        server = get(server_id, workspace=workspace)

        if server.is_running:
            return server

        if server.is_failed:
            raise Exception(f"MLflow server failed: {server.error}")

        time.sleep(poll_interval)

    raise TimeoutError(f"MLflow server {server_id} did not start within {timeout}s")


def get_or_create(
    cluster_name: str,
    workspace: Optional[str] = None,
) -> MLflowServer:
    """
    Get existing MLflow server or create a new one.

    This is a convenience function that returns an existing running server
    if one exists, or creates a new one if not.

    Args:
        cluster_name: Cluster to deploy on if creating
        workspace: Workspace ID (uses default if not specified)

    Returns:
        MLflowServer object (running)

    Example:
        server = corerun.mlflow.get_or_create("my-cluster")
        print(f"MLflow tracking URI: http://corerun-build-{server.id}:5000")
    """
    # Check for existing running server
    servers = list(workspace=workspace)
    running = [s for s in servers if s.is_running]

    if running:
        return running[0]

    # Create new server
    return create(cluster_name, workspace=workspace, wait=True)


def get_tracking_uri(
    server_id: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Optional[str]:
    """
    Get the MLflow tracking URI for the workspace's server.

    Args:
        server_id: Specific server ID (if None, uses first running server)
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Tracking URI string, or None if no server is running

    Note:
        For jobs running on the same cluster, use the in-cluster URL:
        http://corerun-build-{server_id}:5000

        For external access, use the proxy URL through corerun API.

    Example:
        uri = corerun.mlflow.get_tracking_uri()
        if uri:
            mlflow.set_tracking_uri(uri)
    """
    if server_id:
        server = get(server_id, workspace=workspace)
        if server.is_running:
            return f"http://corerun-build-{server.id}:5000"
        return None

    servers = list(workspace=workspace)
    running = [s for s in servers if s.is_running]

    if running:
        return f"http://corerun-build-{running[0].id}:5000"

    return None
