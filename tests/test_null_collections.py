"""The shapes the Go API actually sends, which is where the CLI broke."""
from corerun.registry import RegisteredModel, ModelVersion


def test_a_model_with_no_tags_parses():
    # Verbatim from POST /api/v1/registry/models on a model created with none.
    m = RegisteredModel.model_validate({
        "name": "gitplane-check", "description": "x", "tags": None,
        "latest_version": 0, "version_count": 0,
        "created_by": "e176830c-d48b-4381-8136-98f1fa13d4b6",
        "created_at": "2026-09-17T06:18:27.313Z", "updated_at": "2026-09-17T06:18:27.313Z",
    })
    assert m.tags == {}


def test_a_version_with_no_metadata_parses():
    v = ModelVersion.model_validate({
        "model_name": "gitplane-check", "version": 1, "metadata": None,
        "storage_path": None, "stage": "none",
    })
    assert v.metadata == {}
    assert v.storage_path == ""


def test_a_required_field_is_still_required():
    import pytest
    with pytest.raises(Exception):
        RegisteredModel.model_validate({"name": None})
