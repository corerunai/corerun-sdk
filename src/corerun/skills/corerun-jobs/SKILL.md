---
name: corerun-jobs
description: Submit, monitor, and debug training jobs on corerun — list jobs, read logs, check status, stop or delete them. Use for any request to train a model, run a training script, check why a job failed, or find what is currently running.
---

# corerun jobs

```bash
corerun jobs list                     # everything in the workspace
corerun jobs list --status running
corerun jobs get <job-id>
corerun jobs logs <job-id>            # add --follow to stream
corerun jobs wait <job-id>
corerun jobs cancel <job-id>
```

## Submitting

A job needs an image, a command, and a compute target. Check the target exists
and the workspace has room before submitting:

```bash
corerun compute list                  # valid --compute values
corerun quota show                    # GPU and job headroom
corerun jobs submit --name train --image <image> --compute <target> --gpu 1
```

**Submitting allocates GPUs and costs money.** Confirm the image, command,
dataset and GPU count with the human before submitting anything — and never
submit a speculative job to see what happens.

## Debugging a failure

Read the logs before theorising: `corerun jobs logs <job-id>` returns the tail,
which usually carries the traceback. `corerun jobs get <job-id>` shows the
config it actually ran with, which is where a wrong dataset path or missing
environment variable shows up.

A job that never leaves `pending` is usually waiting on capacity rather than
broken — check `corerun quota gpus` and `corerun clusters list` before
concluding anything about the code.
