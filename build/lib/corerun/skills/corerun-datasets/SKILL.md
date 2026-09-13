---
name: corerun-datasets
description: Find, import, and inspect datasets on corerun — from HuggingFace, Kaggle, URLs or uploads, with schema, format and size. Use when a task needs training data, when asked what data is available, or before wiring a dataset into a job or fine-tune.
---

# corerun datasets

```bash
corerun datasets list
corerun datasets get <dataset-id>       # schema, format, size, storage location
corerun datasets view <dataset-id>      # sample rows
```

## Importing

```bash
corerun datasets import huggingface <repo-id> --name <name>
corerun datasets import kaggle <dataset-id> --name <name>
corerun datasets import url <url> --name <name>
```

Large imports take a while and consume workspace storage — check
`corerun quota show` for storage headroom first, and confirm with the human
before importing anything sizeable.

## Before using one in a job

Check it exists and matches what the training code expects:
`corerun datasets get <id>` shows the real schema and format. A job that fails
immediately on a data path or a column name is nearly always a mismatch here,
so read the schema rather than assuming the shape.
