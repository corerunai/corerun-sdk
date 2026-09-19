---
name: corerun-inference
description: Deploy, scale, and manage model-serving endpoints on corerun — vLLM, SGLang and Ollama servers, LoRA adapters, replica scaling. Use for any request to serve a model, get an OpenAI-compatible endpoint, scale a server up or down, or find out why an endpoint is not responding.
---

# corerun inference servers

```bash
corerun inference list                       # add --status running
corerun inference get <server-id>
corerun inference types                      # available engines
corerun inference wait <server-id>           # blocks until running
```

## Deploying

```bash
corerun quota show                           # GPU and server headroom first
corerun compute list                         # valid --compute values
corerun inference deploy --name <name> --model <model-id> --compute <target> --gpu 1
```

Useful flags: `--type vllm|sglang|ollama`, `--source huggingface|registry|catalog|path`,
`--quantization awq|gptq|fp8`, `--tensor-parallel N`, `--max-model-len N`, `--wait`.

Those four are what vLLM serves. A Triton deployment additionally accepts
artifacts fetched from a tracking server; vLLM does not, and asking for that on
a language model is refused before anything is deployed rather than failing
later inside the container.

`--endpoint <name>` puts the server behind an existing address instead of
creating one named after the deployment, and `--served-name` sets what callers
ask for — needed with `--source path`, which would otherwise publish a
directory as the model name. `--arg` passes an engine flag through as given,
repeated per token: `--arg --kv-cache-dtype --arg fp8`.

The address, its keys, and publishing models somebody else runs are the
[corerun-endpoints](../corerun-endpoints/SKILL.md) skill.

**Deploying and scaling up allocate GPUs and hold them until stopped** — an
endpoint is a standing cost, unlike a job that finishes. Confirm the model,
engine, GPU count and compute target with the human before deploying, and never
deploy a second copy of something already running: check `corerun inference list`
first.

## Scaling and stopping

```bash
corerun inference scale <server-id> --min-replicas 2 --max-replicas 4
corerun inference stop <server-id>           # releases GPUs, keeps the record
corerun inference delete <server-id>
```

Scaling up allocates more GPUs; check quota first. **Stopping or deleting an
endpoint someone is using is an outage** — confirm before either, and never do
it to free capacity for your own work.

## When an endpoint does not answer

`corerun inference get <server-id>` shows status and the endpoint URL. A
server in `pending` or `deploying` is still pulling weights, which takes minutes
for a large model — `corerun inference wait` rather than concluding it is broken.
A `failed` status carries an error field; read it before retrying.

`running` means the port answered, not merely that a container started. A
server that reports running and then fails a request is a different problem
from one that never came up: read the engine's own output rather than the
status.
