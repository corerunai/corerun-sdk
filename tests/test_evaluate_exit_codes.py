"""What a pipeline actually depends on: the exit code.

A build that cannot tell "the model got worse" from "the run crashed" gets
ignored within a week, so these pin the three outcomes apart.
"""
import json
from types import SimpleNamespace

import pytest
import typer

import corerun.cli.evaluate as ev_cli


def _run(status="completed", scores=None, error=None):
    return SimpleNamespace(
        status=status, progress=100, error=error,
        results={"aggregate_scores": scores or {}},
    )


def _invoke(monkeypatch, run, **kwargs):
    monkeypatch.setattr(ev_cli, "_init_client", lambda *a, **k: None)
    import corerun.evaluations as ev
    monkeypatch.setattr(ev, "wait_run", lambda *a, **k: run)
    defaults = dict(run_id="r1", timeout=None, fail_under=None, as_json=False, workspace=None)
    defaults.update(kwargs)
    return ev_cli.wait_for_run(**defaults)


def test_a_passing_run_exits_zero(monkeypatch):
    _invoke(monkeypatch, _run(scores={"correctness": 0.81}))


def test_a_failed_run_exits_two_not_zero(monkeypatch):
    """It used to print a green tick and exit 0 for a failed run, which is why
    an evaluation step in CI passed no matter what happened."""
    with pytest.raises(typer.Exit) as e:
        _invoke(monkeypatch, _run(status="failed", error="engine died"))
    assert e.value.exit_code == 2


def test_scoring_below_the_bar_exits_one(monkeypatch):
    with pytest.raises(typer.Exit) as e:
        _invoke(monkeypatch, _run(scores={"correctness": 0.41}), fail_under=0.6)
    assert e.value.exit_code == 1


def test_a_low_score_is_distinguishable_from_a_broken_run(monkeypatch):
    """1 and 2 must not collapse: one is a result, the other is an outage."""
    with pytest.raises(typer.Exit) as low:
        _invoke(monkeypatch, _run(scores={"c": 0.1}), fail_under=0.5)
    with pytest.raises(typer.Exit) as broken:
        _invoke(monkeypatch, _run(status="failed"), fail_under=0.5)
    assert low.value.exit_code != broken.value.exit_code


def test_no_threshold_means_no_opinion(monkeypatch):
    _invoke(monkeypatch, _run(scores={"correctness": 0.01}))


def test_json_is_parseable(monkeypatch, capsys):
    _invoke(monkeypatch, _run(scores={"correctness": 0.9}), as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["scores"]["correctness"] == 0.9
