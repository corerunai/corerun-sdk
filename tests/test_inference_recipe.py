"""The recipe a model's publisher recommends, as the SDK and CLI report it.

The point of reading one is the flag nobody knows to set -- a tool-call parser
above all -- so what these pin is that the arguments survive the trip from the
API to the terminal, and that having none is not an error.
"""
import corerun.cli.inference as inf_cli
import corerun.inference as inference


class _Response(dict):
    pass


class _Client:
    """Stands in for the HTTP client, recording what was asked of it."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, path, params=None, workspace=None):
        self.calls.append((path, params, workspace))
        return self.response


def _patch_client(monkeypatch, response):
    client = _Client(response)
    monkeypatch.setattr(inference, "get_client", lambda: client)
    return client


def test_the_arguments_come_through(monkeypatch):
    client = _patch_client(
        monkeypatch,
        {
            "model": "Inferact/Qwen3.8-27B-NVFP4",
            "found": True,
            "recipe": {
                "hf_id": "Inferact/Qwen3.8-27B-NVFP4",
                "title": "Qwen3.8-27B",
                "provider": "Qwen",
                "args": ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"],
                "image": "vllm/vllm-openai:qwen38",
                "min_vllm_version": "0.17.0",
                "context_length": 262144,
                "hardware": "b200",
                "source": "https://recipes.vllm.ai/Inferact/Qwen3.8-27B-NVFP4.json",
            },
        },
    )

    r = inference.recipe("Inferact/Qwen3.8-27B-NVFP4")

    assert r.found
    assert r.args[-2:] == ["--tool-call-parser", "qwen3_coder"]
    assert r.context_length == 262144
    # The model is asked about by name, and the workspace travels with it.
    assert client.calls == [("/inference-servers/recipe", {"model": "Inferact/Qwen3.8-27B-NVFP4"}, None)]


def test_a_model_with_no_recipe_is_an_answer_not_a_failure(monkeypatch):
    _patch_client(
        monkeypatch,
        {"model": "someone/Nowhere-9B", "found": False, "reason": "recipes: nothing published"},
    )

    r = inference.recipe("someone/Nowhere-9B")

    assert not r.found
    assert r.args == []
    assert "nothing published" in r.reason


def test_a_missing_recipe_field_does_not_crash_the_wrapper(monkeypatch):
    # The API can answer with no recipe body at all; a client that index-errors
    # on that is worse than one that says it does not know.
    _patch_client(monkeypatch, {"model": "x", "found": False})

    assert inference.recipe("x").found is False


def test_the_command_prints_the_arguments(monkeypatch, capsys):
    monkeypatch.setattr(inf_cli, "_init_client", lambda *a, **k: None)
    monkeypatch.setattr(
        inference,
        "recipe",
        lambda model, workspace=None: inference.ServingRecipe(
            model_id=model,
            found=True,
            hf_id="Qwen/Qwen3.8-27B",
            title="Qwen3.8-27B",
            args=["--tool-call-parser", "qwen3_coder"],
            hardware="b200",
        ),
    )

    inf_cli.model_recipe("Qwen/Qwen3.8-27B", workspace=None, json_output=False)

    out = capsys.readouterr().out
    assert "--tool-call-parser" in out
    assert "qwen3_coder" in out
    assert "b200" in out


def test_the_command_says_so_when_there_is_none(monkeypatch, capsys):
    monkeypatch.setattr(inf_cli, "_init_client", lambda *a, **k: None)
    monkeypatch.setattr(
        inference,
        "recipe",
        lambda model, workspace=None: inference.ServingRecipe(
            model_id=model, found=False, reason="nothing published for it"
        ),
    )

    inf_cli.model_recipe("someone/Nowhere-9B", workspace=None, json_output=False)

    out = capsys.readouterr().out
    assert "No serving recipe published" in out
    assert "nothing published for it" in out
