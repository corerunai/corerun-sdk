---
name: corerun-quota
description: Check what the workspace has room for before allocating anything on corerun — GPUs, jobs, notebooks, inference servers, storage. Use before submitting a job, starting a fine-tune, or deploying an endpoint, and whenever a creation fails with a limit or quota error.
---

# corerun quota

Creation is refused once a limit is reached. Check first — a refused job wastes
a round trip and an exhausted GPU pool wastes everyone's.

```bash
corerun quota show            # every limit, with usage and headroom
corerun quota gpus            # GPUs broken down by the workload holding them
corerun quota show --json     # for parsing
```

`show` flags anything already at its limit under **At limit:**. A limit of
`unlimited` has no ceiling.

## Before allocating

Run `corerun quota show` and compare against what you are about to ask for.
GPUs are the contended resource: a job asking for 4 when 2 are free will not
schedule, and nothing will tell you why until it sits queued.

If a resource is at its limit, say so and stop. Do not free room by stopping
someone else's work — jobs and endpoints belong to people, and a stopped
endpoint is an outage. Report what is full and let the human decide.

## When creation fails

A `quota` or `limit` error means the ceiling, not a bug. Run `corerun quota show`,
report which resource is exhausted and what currently holds it
(`corerun quota gpus` for GPUs, `corerun jobs list` for jobs), and stop.
