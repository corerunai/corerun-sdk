---
name: corerun-workspaces
description: Create and delete corerun workspaces, choose which one the CLI acts in, and manage the organization's storage accounts and compute. Use when asked to set up a new workspace, switch between workspaces, add an S3 or ObjectIO account, or connect a cluster or bare-metal host.
---

# Workspaces, storage and compute

These are the administrative commands: they change what exists for everyone in
the organization, not what runs inside one workspace. Most need an
organization administrator.

## Which workspace you are in

Every resource command acts in one workspace, sent as a header. Check before
doing anything that allocates.

```bash
corerun workspace show          # which one commands act in
corerun workspace list          # the ones you belong to
corerun workspace set speech    # change it
```

`corerun ws` is the shorter spelling of all of these.

## Creating a workspace

```bash
corerun ws create "ML Research"
corerun ws create "Speech" --slug speech --capabilities notebooks,training
corerun ws create "Scratch" --use
```

The slug is derived from the name when omitted. `--use` makes it the workspace
this CLI acts in, which saves a `set` afterwards.

`--capabilities` is a subset of `notebooks, training, models,
datasets, images, endpoints`; omitting it means all of them. It decides what
the console shows and what the workspace is for — a serving-only workspace
asking for `endpoints` is clearer than one that offers training nobody will
use. It is not an authorization boundary: do not reach for it to stop somebody
doing something.

Storage comes from the organization's shared account when there is one. The
workspace gets a bucket of its own — `corerun-<slug>` — and on ObjectIO a
credential confined to that bucket. Nothing to pass; it happens on creation.
**A workspace created before the organization had a shared account does not get
one retroactively.** Add storage first.

## Deleting a workspace

```bash
corerun ws delete scratch
corerun ws delete scratch --yes
```

**This deletes everything in it** — notebooks, jobs, endpoints, registered
models, datasets. Name, slug or ID all resolve. Confirm with the human first
and say what the workspace contains; `--yes` exists for scripts, not for
skipping the question on someone's behalf.

Note it does not currently deprovision the workspace's object-store bucket or
its scoped credential — those outlive the workspace and need removing by hand.

## Storage accounts

Organization-wide by default, which is almost always what is wanted: every
workspace created afterwards draws a bucket from it.

```bash
corerun storage list
corerun storage add orgs3 --endpoint https://s3.example.com \
    --access-key AKIA... --secret-key ...
corerun storage update orgs3 --plane git
corerun storage delete orgs3
```

`--provider` is one of `objectio, minio, ceph, aws, other` (default
`objectio`). Only ObjectIO can mint a credential confined to a single bucket;
on everything else each workspace uses the account key, so the bucket is a
convention rather than a boundary — say so rather than implying isolation.

`--plane` decides where model weights live: `git` (default) puts them in the
platform's git service with LFS, `s3` puts them in the bucket.

`-w/--workspace` scopes an account to the current workspace instead of the
organization. Reach for it only when one workspace genuinely needs its own
account; the org-wide one is what makes new workspaces work without setup.

**Changing or removing an account does not move anything.** Datasets,
checkpoints and model weights stay in the old account and stop being readable.
Treat `update` of an endpoint or key, and `delete`, as breaking, and say so
before running either.

## Compute

```bash
corerun clusters list
corerun clusters get <name>
corerun clusters profiles <name>       # the shapes people can launch
corerun clusters add <name>            # prints the manifest to apply
corerun clusters rm <name>
corerun hosts add <name>               # bare metal; prints an installer
corerun hosts rm <name>
```

`clusters list` returns both the organization's shared clusters and this
workspace's own, each tagged with its scope. A cluster added at organization
scope is shared into every workspace; one added at workspace scope is not.
Check the scope before removing anything — removing a shared cluster takes it
away from workspaces you are not looking at.

`clusters add` and `hosts add` do not reach out to the machine. They hand back
a manifest or an installer for somebody to run there, and the operator joins
from its side. Nothing runs until it does.
