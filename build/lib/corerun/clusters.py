"""
corerun clusters.

Read-only access to the clusters a workspace can schedule onto, including live
resource data reported by each cluster's connector.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from corerun.config import get_client


class GPUInfo(BaseModel):
    """A GPU model present on a cluster, with how many of them there are."""

    vendor: str = ""
    model: str = ""
    count: int = 0


class StorageClassInfo(BaseModel):
    """A storage class workloads on this cluster can request."""

    name: str = ""
    provisioner: str = ""
    is_default: bool = False


class ClusterResources(BaseModel):
    """Capacity discovered on a cluster, and how much of it is spoken for.

    Present only while the cluster's connector is connected; a cluster that has
    never connected reports no resources at all.
    """

    node_count: int = 0
    total_cpu: int = 0
    total_memory_mb: int = 0
    total_gpus: int = 0
    allocated_cpu: int = 0
    allocated_memory_mb: int = 0
    allocated_gpus: int = 0
    gpus: List[GPUInfo] = Field(default_factory=list)
    storage_classes: List[StorageClassInfo] = Field(default_factory=list)
    discovered_at: Optional[str] = None

    @property
    def available_gpus(self) -> int:
        """GPUs not currently allocated to a running workload."""
        return max(self.total_gpus - self.allocated_gpus, 0)


class Cluster(BaseModel):
    """A cluster registered with the platform."""

    id: str = ""
    name: str = ""
    backend: str = ""
    namespace: str = ""
    status: str = ""
    scope: str = ""
    initialized: bool = False
    connector_connected: bool = False
    gpu_strategy: str = ""
    architecture: Optional[str] = None
    cluster_type: Optional[str] = None
    default_profile: Optional[str] = None
    resources: Optional[ClusterResources] = None

    @property
    def is_ready(self) -> bool:
        """Whether the cluster can accept work right now."""
        return self.status == "ready" and self.connector_connected

    def __repr__(self) -> str:
        return f"<Cluster {self.name} ({self.status})>"


def _from_response(data: dict) -> Cluster:
    return Cluster.model_validate(data)


def list(workspace: Optional[str] = None) -> List[Cluster]:
    """
    List the clusters available to the workspace.

    Includes both tenant-wide clusters and clusters scoped to this workspace.

    Args:
        workspace: Workspace ID (uses default if not specified)

    Returns:
        List of Cluster objects

    Example:
        for c in corerun.clusters.list():
            print(c.name, c.status)
    """
    client = get_client()
    response = client.get("/clusters", workspace=workspace)
    return [_from_response(c) for c in response.get("clusters", [])]


def get(name: str, workspace: Optional[str] = None) -> Cluster:
    """
    Get one cluster by name.

    Args:
        name: Cluster name
        workspace: Workspace ID (uses default if not specified)

    Returns:
        Cluster object, including live resources when its connector is connected

    Example:
        c = corerun.clusters.get("gb10dgx01")
        if c.resources:
            print(c.resources.available_gpus, "GPUs free")
    """
    client = get_client()
    return _from_response(client.get(f"/clusters/{name}", workspace=workspace))


def profiles(name: str, workspace: Optional[str] = None) -> list:
    """
    List the pod profiles (CPU/memory/GPU presets) a cluster offers.

    Args:
        name: Cluster name
        workspace: Workspace ID (uses default if not specified)

    Returns:
        List of profile dicts
    """
    client = get_client()
    response = client.get(f"/clusters/{name}/profiles", workspace=workspace)
    if isinstance(response, dict):
        return response.get("profiles", [])
    return response
