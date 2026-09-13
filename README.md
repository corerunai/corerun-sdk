# corerun Python SDK

Python SDK and CLI for the corerun ML Platform.

## Installation

```bash
uv tool install "git+https://github.com/corerunai/corerun-sdk.git"
```

`pip install "git+https://github.com/corerunai/corerun-sdk.git"` does the same.
Not on PyPI yet; the git URL is the install until it is.

From a checkout:

```bash
cd corerun-sdk
pip install -e .
```

## Pointing it somewhere

It talks to `https://corerun.ai/api/v1` unless told otherwise, and reads model
endpoints from `https://api.corerun.ai`. A self-hosted deployment overrides both
in `~/.corerun/config`:

```
api_url=https://corerun.example.com/api/v1
inference_url=https://api.corerun.example.com
```

`CORERUN_API_URL` and `CORERUN_INFERENCE_URL` override the file, which is what
CI uses. `corerun login` writes the file for you.

## Quick Start

### Authentication

Log in once, from a terminal:

```bash
corerun login
```

That opens a browser, and writes the token and workspace to `~/.corerun/config`.
Everything after it — the CLI and the SDK alike — reads that file, so nothing
needs a credential passed to it:

```python
import corerun

corerun.init()          # uses ~/.corerun/config
```

Where there is no browser — a container, a remote box, a notebook on a cluster
— log in the same way and approve it from a machine that has one:

```bash
corerun login --use-device-code
```

It prints a code, you approve it in a browser anywhere, and it writes the same
`~/.corerun/config`. Everything after that works as above, still with no
credential in your code.

A token is for the case where nobody is present to approve anything — a CI job,
a cron. Put it in the environment rather than in a call, so it does not end up
in a file somebody commits:

```bash
export CORERUN_AUTH_TOKEN=cr-xxx
export CORERUN_WORKSPACE=your-workspace-id
```


### Datasets

```python
import corerun

# List datasets
datasets = corerun.datasets.list()
for ds in datasets:
    print(f"{ds.name}: {ds.size_mb:.1f}MB")

# Import from HuggingFace
ds = corerun.datasets.from_huggingface("ylecun/mnist", name="mnist")

# Import from Kaggle
ds = corerun.datasets.from_kaggle("uciml/iris", name="iris")

# View dataset rows
viewer = corerun.datasets.view("mnist", page=1, page_size=10)
print(f"Total rows: {viewer.total_rows}")
for row in viewer.rows:
    print(row)

# Get dataset info
ds = corerun.datasets.get("mnist")
print(f"Size: {ds.size_mb:.1f}MB, Files: {ds.file_count}")

# Delete dataset
corerun.datasets.delete("old-dataset")
```

### Jobs

```python
import corerun

# Submit a training job
job = corerun.jobs.submit(
    name="train-model",
    image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    compute_name="dgx-cluster",
    command=["python", "train.py"],
    args=["--epochs", "100"],
    gpu=1,
    datasets=["mnist"],
    environment={"WANDB_API_KEY": "xxx"},
)

print(f"Job submitted: {job.id}")

# Wait for completion
job = corerun.jobs.wait(job.id)
print(f"Job finished with status: {job.status}")

# Get logs
logs = corerun.jobs.logs(job.id)
print(logs)

# List running jobs
running = corerun.jobs.list(status="running")

# Cancel a job
corerun.jobs.cancel(job.id)

# Run and wait (convenience function)
job = corerun.jobs.run(
    name="quick-task",
    image="python:3.11",
    compute_name="cpu-cluster",
    command=["python", "-c", "print('Hello')"],
)
```

### Jobs with Local Training Code

Upload local training code to run on remote compute clusters:

```python
import corerun

# Submit job with local source code
job = corerun.jobs.submit(
    name="train-custom-model",
    image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    compute_name="dgx-cluster",
    source_directory="./src",  # Upload local code
    command=["python", "train.py"],
    gpu=1,
    datasets=["mnist"],
    experiment="my-experiment",
)

# Code is uploaded to storage and mounted at /code
# Working directory is automatically set to /code
# CORERUN_CODE_DIR=/code is set in environment

# Run and wait for completion
job = corerun.jobs.run(
    name="train-and-wait",
    image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
    compute_name="dgx-cluster",
    source_directory="./training",
    command=["python", "train.py", "--epochs", "10"],
    gpu=1,
    timeout=3600,  # 1 hour timeout
)
print(f"Training completed with status: {job.status}")
```

Files excluded by default when uploading source code:

- `__pycache__`, `*.pyc`, `.git`, `.venv`, `venv`, `node_modules`
- `.env`, `*.log`, `.coverage`, `dist`, `build`

Override with `exclude_patterns`:

```python
job = corerun.jobs.submit(
    name="train",
    image="python:3.11",
    compute_name="cpu-cluster",
    source_directory="./src",
    exclude_patterns=["*.pyc", "__pycache__", "tests"],  # Custom patterns
    command=["python", "train.py"],
)
```

### Model Registry

```python
import corerun

# List registered models
models = corerun.registry.list_models()
for model in models:
    print(f"{model.name}: {model.version_count} versions")

# Create a new model
model = corerun.registry.create_model(
    "resnet50",
    description="ResNet-50 image classifier",
    tags={"framework": "pytorch", "task": "classification"},
)

# Upload a model file
version = corerun.registry.upload_model(
    "resnet50",
    local_path="model.pt",
    framework="pytorch",
    description="Trained on ImageNet",
)
print(f"Uploaded as v{version.version}")

# Or log a model object directly
import torch
model = torch.load("model.pt")
version = corerun.registry.log_model(
    "resnet50",
    model,
    framework="pytorch",
)

# Transition to production
corerun.registry.transition_stage("resnet50", version.version, "production")

# Set aliases
corerun.registry.set_alias("resnet50", "champion", version.version)
corerun.registry.set_alias("resnet50", "latest", version.version)

# Download a model by version or alias
corerun.registry.download_model("resnet50", "model.pt", version=1)
corerun.registry.download_model("resnet50", "model.pt", alias="champion")

# Load model directly
model = corerun.registry.load_model("resnet50", alias="champion", framework="pytorch")
model.eval()

# List versions
versions = corerun.registry.list_versions("resnet50")
for v in versions:
    print(f"v{v.version}: {v.stage} ({v.size_mb:.1f}MB)")

# List aliases
aliases = corerun.registry.list_aliases("resnet50")
for alias in aliases:
    print(f"@{alias.alias} -> v{alias.version}")

# Delete a model version
corerun.registry.delete_version("resnet50", 1)

# Delete entire model
corerun.registry.delete_model("old-model")
```

### Model Inference

```python
import corerun

# Quick one-off prediction
result = corerun.registry.predict(
    "my-model",
    input_data,
    alias="champion",
)

# For multiple predictions, use a predictor to avoid reloading
predictor = corerun.registry.get_predictor("my-model", alias="champion")

# Single prediction
result = predictor.predict(input_data)

# Batch prediction
results = predictor.predict_batch([input1, input2, input3], batch_size=16)

# Get probabilities (classifiers)
probs = predictor.predict_proba(input_data)

# With preprocessing and postprocessing
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

predictor = corerun.registry.get_predictor(
    "resnet50",
    alias="champion",
    preprocess=lambda img: transform(img).unsqueeze(0),
    postprocess=lambda out: out.argmax(dim=1).item(),
    device="cuda",  # Optional: specify device
)

# Predict on an image
label = predictor.predict(pil_image)

# Batch predict
labels = predictor.predict_batch(images, batch_size=32)
```

## CLI Usage

### Authentication

```bash
# Login (saves credentials to ~/.corerun/config)
corerun login --workspace your-workspace-id

# Interactive login
corerun login

# Check status
corerun whoami

# Logout
corerun auth logout
```

### Datasets

```bash
# List datasets
corerun datasets list

# Get dataset details
corerun datasets get mnist

# View dataset rows
corerun datasets view mnist --split train --page 1 --size 10

# Import from HuggingFace
corerun datasets import huggingface ylecun/mnist
corerun datasets import huggingface ylecun/mnist --name my-mnist

# Import from Kaggle
corerun datasets import kaggle uciml/iris --name iris

# Import from URL
corerun datasets import url https://example.com/data.zip --name my-data

# Delete dataset
corerun datasets delete old-dataset
corerun datasets delete old-dataset --force
```

### Jobs

```bash
# List jobs
corerun jobs list
corerun jobs list --status running
corerun jobs list --compute dgx-cluster

# Submit a job
corerun jobs submit \
    --name train-model \
    --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime \
    --compute dgx-cluster \
    --gpu 1 \
    --datasets mnist \
    --command "python train.py"

# Submit with local training code
corerun jobs submit \
    --name train-custom \
    --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime \
    --compute dgx-cluster \
    --source ./src \
    --command "python train.py" \
    --gpu 1

# Submit and wait
corerun jobs submit \
    --name train \
    --image python:3.11 \
    --compute cpu-cluster \
    --command "python train.py" \
    --wait

# Get job details
corerun jobs get abc123

# Get logs
corerun jobs logs abc123
corerun jobs logs abc123 --tail 100
corerun jobs logs abc123 --follow

# Wait for job
corerun jobs wait abc123
corerun jobs wait abc123 --timeout 3600

# Cancel job
corerun jobs cancel abc123

# Delete job
corerun jobs delete abc123
```

### Model Registry

```bash
# List models
corerun models list
corerun models list --json

# Get model details
corerun models get my-model

# Create a new model
corerun models create my-model --description "My ML model"
corerun models create my-model -t framework=pytorch -t task=classification

# Upload a model file
corerun models upload my-model model.pt --framework pytorch
corerun models upload my-model model.pt -d "Trained for 100 epochs"

# Download a model
corerun models download my-model model.pt --version 1
corerun models download my-model model.pt --alias champion

# List versions
corerun models versions my-model

# Transition stage
corerun models stage my-model 1 staging
corerun models stage my-model 1 production
corerun models stage my-model 1 archived

# Manage aliases
corerun models alias my-model champion 1
corerun models alias my-model challenger 2
corerun models aliases my-model
corerun models unalias my-model challenger

# Delete model
corerun models delete old-model
corerun models delete old-model --force

# Run inference
corerun models predict my-model input.json --alias champion
corerun models predict my-model image.jpg --version 1 -o result.json
corerun models predict my-model data.csv --alias production --json
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CORERUN_AUTH_TOKEN` | Auth token, for when a browser login is not possible |
| `CORERUN_WORKSPACE` | Default workspace ID |
| `CORERUN_API_URL` | API base URL (default: https://dev.corerun.ai/api/v1) |
| `CORERUN_TIMEOUT` | Request timeout in seconds (default: 30) |
| `CORERUN_AUTH_TOKEN_FILE` | Where to read the platform credential (default: `/etc/corerun/api-key`) |

### Inside a corerun notebook

A notebook is launched with a credential of its own at `/etc/corerun/api-key`,
so the SDK and CLI work with no setup. The key is short-lived and the platform
replaces the file every few hours while the notebook runs — the SDK reads it
when it needs it, and re-reads it after a 401, so a renewal is invisible.

It is the last source consulted, so anything you choose yourself — `corerun
login`, `CORERUN_AUTH_TOKEN`, an explicit `api_key=` — keeps your own identity.

### Config File

Credentials can be saved to `~/.corerun/config`:

```ini
api_key=cr-xxx
workspace=your-workspace-id
api_url=https://dev.corerun.ai/api/v1
timeout=30
```

## Error Handling

```python
from corerun.exceptions import (
    CoreRunError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
)

try:
    ds = corerun.datasets.get("nonexistent")
except NotFoundError:
    print("Dataset not found")
except AuthenticationError:
    print("Invalid API key")
except CoreRunError as e:
    print(f"API error: {e}")
```

## License

Apache-2.0
