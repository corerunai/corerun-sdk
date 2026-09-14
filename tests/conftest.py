"""
Shared test setup.

The CLI keeps its output mode in module state rather than passing it around:
`output.set_json` flips a global and repoints the one Console every module
holds a reference to. That is the right shape for a command-line tool, which
runs one command and exits -- and the wrong shape for a test process, where it
outlives the test that set it.

Without this, a test that runs something with `--json` leaves every test after
it believing it is still in JSON mode, with the console aimed at a stream the
test harness has already closed. The failures that produces land in whichever
test happens to run next, which is why this is reset here rather than in the
files that use the flag.
"""

import pytest

from corerun.cli import output


@pytest.fixture(autouse=True)
def _restore_cli_output(monkeypatch):
    monkeypatch.setattr(output, "_json_mode", False)
    monkeypatch.setattr(output, "_emitted", False)
    # None, not sys.stdout: that is the Console's own default and means "the
    # stdout in force when something is written", which is what set_json
    # restores on its way out.
    monkeypatch.setattr(output.console, "file", None)
    yield
