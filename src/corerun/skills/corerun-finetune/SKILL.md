---
name: corerun-finetune
description: Run LoRA and QLoRA fine-tuning jobs on corerun with Unsloth or HuggingFace — create fine-tunes, pick base models and datasets, track them to completion. Use when asked to fine-tune, adapt, or specialise a model on a dataset.
---

# corerun fine-tuning

```bash
corerun finetune list                    # add --status running or --framework unsloth
corerun finetune create --name <name> --framework unsloth \
    --model <base-model> --dataset <dataset-id> --compute <target>
corerun finetune wait <job-id>
```

Frameworks: `unsloth` (4-bit quantised, cheapest on GPU memory) and
`huggingface` (HF Trainer). Tunables include epochs, batch size, learning rate,
LoRA rank and alpha, and max sequence length.

## Before creating

Three things fail a fine-tune before it starts, so check them:

```bash
corerun quota show                       # GPUs available
corerun datasets get <dataset-id>        # schema matches what the framework wants
corerun compute list                     # target exists and has the right GPUs
```

**A fine-tune holds GPUs for hours.** Confirm base model, dataset, framework
and GPU count with the human before creating one, and check
`corerun finetune list` for an equivalent run already going — duplicated
fine-tunes are the most expensive mistake available here.

## Afterwards

A finished fine-tune produces a LoRA adapter that
[corerun-inference](../corerun-inference/SKILL.md) can load onto a base model,
and that [corerun-models](../corerun-models/SKILL.md) can register and stage.
