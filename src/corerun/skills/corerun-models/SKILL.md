---
name: corerun-models
description: Work with the corerun model registry — list and create registered models, upload and download versions, stage them through Development/Staging/Production, and set aliases like champion. Use when asked to register, promote, fetch or version a model.
---

# corerun model registry

```bash
corerun models list
corerun models versions <name>
corerun models download <name> <file> --alias champion
corerun models create <name> --framework <framework>
corerun models upload <name> <file> --framework <framework>
```

## Staging and aliases

```bash
corerun models stage <name> <version> production
corerun models alias <name> champion <version>
```

Stages run Development → Staging → Production → Archived. An alias is a moving
pointer, so `champion` can be repointed without callers changing anything —
prefer aliases over version numbers when wiring a model into a serving path.

**Promoting to production changes what live traffic gets.** Confirm the version
with the human, and say which version the alias moved from and to. Never
promote a version you have not seen evaluated.
