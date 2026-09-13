"""
corerun CLI

Command-line interface for corerun ML Platform.

Usage:
    corerun login
    corerun datasets list
    corerun datasets import huggingface ylecun/mnist --name mnist
    corerun jobs submit --name train --image pytorch/pytorch:latest --gpu 1
    corerun jobs logs <job-id>
    corerun models list
    corerun models upload my-model model.pt --framework pytorch
    corerun models download my-model model.pt --alias champion
    corerun models stage my-model 1 production
    corerun models alias my-model champion 1
    corerun models predict my-model input.json --alias champion
    corerun prompts list
    corerun prompts get my-prompt --alias production
    corerun prompts create my-prompt --template "Hello {{name}}"
    corerun prompts alias my-prompt production 2
    corerun traces list
    corerun traces get <trace-id>
    corerun finetune list
    corerun finetune create --name my-ft --framework unsloth --model meta-llama/Llama-3.2-3B-Instruct --dataset <id> --compute dgx
    corerun finetune wait <job-id>
    corerun evaluate scorers
    corerun evaluate datasets list
    corerun evaluate datasets create --name qa-bench --schema schema.json --examples data.json
    corerun evaluate runs list
    corerun evaluate runs create --name my-eval --dataset <id> --model mistral-7b --endpoint <server-id> --scorers correctness,fluency --compute dgx --wait
    corerun inference list
    corerun inference deploy --name mistral --model mistralai/Mistral-7B --compute dgx --gpu 1
    corerun inference scale <server-id> --min-replicas 2 --max-replicas 4
    corerun clusters list
    corerun clusters get gb10dgx01
    corerun compute list
    corerun quota show
"""

import typer
from rich.console import Console

from corerun.cli.auth import app as auth_app
from corerun.cli.clusters import app as clusters_app
from corerun.cli.compute import app as compute_app
from corerun.cli.datasets import app as datasets_app
from corerun.cli.endpoints import app as endpoints_app
from corerun.cli.evaluate import app as evaluate_app
from corerun.cli.finetune import app as finetune_app
from corerun.cli.inference import app as inference_app
from corerun.cli.jobs import app as jobs_app
from corerun.cli.notebooks import app as notebooks_app
from corerun.cli.prompts import app as prompts_app
from corerun.cli.quota import app as quota_app
from corerun.cli.registry import app as registry_app
from corerun.cli.skills import app as skills_app
from corerun.cli.workspace import app as workspace_app
from corerun.cli import output
from corerun.cli.traces import app as traces_app

console = Console()

app = typer.Typer(
    name="corerun",
    help="corerun ML Platform CLI",
    no_args_is_help=True,
    add_completion=True,
)


@app.callback()
def main(
    json_output: bool = typer.Option(
        False, "--json", help="Print results as JSON, for piping into other tools"
    ),
):
    """corerun ML Platform CLI."""
    # Set before any command runs, so a command need only hand its result to
    # output.emit and never branch on the mode itself.
    output.set_json(json_output)

# Add subcommands
app.add_typer(auth_app, name="auth", help="Authentication commands")
app.add_typer(datasets_app, name="datasets", help="Dataset management")
app.add_typer(jobs_app, name="jobs", help="Job management")
app.add_typer(notebooks_app, name="notebooks", help="Notebook sessions")
app.add_typer(registry_app, name="models", help="Model registry")
app.add_typer(prompts_app, name="prompts", help="Prompt registry")
app.add_typer(skills_app, name="skills", help="corerun skills for coding agents")
app.add_typer(traces_app, name="traces", help="Trace management")
app.add_typer(finetune_app, name="finetune", help="Fine-tuning jobs")
app.add_typer(evaluate_app, name="evaluate", help="LLM evaluations")
app.add_typer(inference_app, name="inference", help="Inference servers")
app.add_typer(endpoints_app, name="endpoints", help="Model endpoints")
app.add_typer(clusters_app, name="clusters", help="Cluster inspection")
app.add_typer(compute_app, name="compute", help="Compute targets")
app.add_typer(quota_app, name="quota", help="Workspace quota")
app.add_typer(workspace_app, name="workspace", help="Workspace selection")

# Aliases for convenience
app.add_typer(datasets_app, name="data", hidden=True)
app.add_typer(datasets_app, name="ds", hidden=True)
app.add_typer(registry_app, name="registry", hidden=True)
# Singular reads better for the commands that act on one model ("model pull"),
# and is what people type; both names reach the same commands.
app.add_typer(registry_app, name="model", hidden=True)
app.add_typer(finetune_app, name="ft", hidden=True)
app.add_typer(evaluate_app, name="eval", hidden=True)
app.add_typer(inference_app, name="serve", hidden=True)
app.add_typer(clusters_app, name="cluster", hidden=True)
# "ws" is what gets typed; both reach the same commands.
app.add_typer(workspace_app, name="ws", hidden=True)


@app.command()
def login(
    auth_token: str = typer.Option(None, "--token", "-t", help="Auth token"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Default workspace ID"),
    api_url: str = typer.Option(None, "--url", help="API URL"),
    use_device_code: bool = typer.Option(
        False,
        "--use-device-code",
        help="Show a code to approve elsewhere instead of opening a browser",
    ),
):
    """
    Login to corerun (alias for 'corerun auth login')
    """
    from corerun.cli.auth import login as auth_login
    auth_login(
        auth_token=auth_token,
        workspace=workspace,
        api_url=api_url,
        use_device_code=use_device_code,
    )


@app.command()
def whoami():
    """
    Show current user info (alias for 'corerun auth whoami')
    """
    from corerun.cli.auth import whoami as auth_whoami
    auth_whoami()


@app.command()
def version():
    """
    Show version information
    """
    from corerun import __version__
    console.print(f"corerun version {__version__}")


if __name__ == "__main__":
    app()
