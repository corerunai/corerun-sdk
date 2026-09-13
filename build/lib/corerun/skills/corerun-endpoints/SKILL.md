---
name: corerun-endpoints
description: Model endpoints on corerun — the stable address an application calls, its API keys, and publishing models behind it including ones corerun does not run. Use when asked for the URL to call a model, to rotate or find a key, to put a provider's model behind the platform's address, or when a client gets 401, 404 or the wrong model back.
---

# corerun model endpoints

An endpoint is the address; the models behind it come and go. That is the whole
point of the separation — the URL in somebody's application keeps working while
a model is redeployed, replaced, or moved to another machine.

```bash
corerun endpoints list
corerun endpoints show <name>                # address, keys, what answers
corerun endpoints models <name>              # ask the endpoint itself
corerun endpoints call <name> "prompt" --stream
```

## The address

```
https://<inference-host>/<endpoint-name>
```

Read it from `corerun endpoints show` rather than assembling it. The host is
deployment configuration and differs between environments; a URL built by hand
is one that works until somebody moves it.

## Keys

Two live at once, deliberately:

```bash
corerun endpoints rotate-key <name> --key secondary
```

One key cannot be rotated safely — changing it and changing every caller are the
same instant, and the gap is an outage. Regenerate the idle one, move callers
onto it, then regenerate the other.

**Rotating a key that callers are using is an outage.** Ask which one is idle
before touching either. Do not print a key into a transcript, an issue, or a
commit; `show` masks them for that reason.

## Publishing a model

A deployment joins an endpoint when it is created:

```bash
corerun inference deploy --name <server> --model <id> --compute <target> \
  --endpoint <endpoint-name> --served-name <what-callers-ask-for>
```

A model somebody else runs is published the same way and answers on the same
address:

```bash
corerun endpoints add-upstream <name> --model chat \
  --base-url https://api.openai.com/v1 --api-key sk-... \
  --as gpt-4o --provider openai
corerun endpoints remove-upstream <name> chat
```

`--as` is what the provider is asked for when it differs from the
name callers use — that is what lets a caller keep asking for `chat` while the
model behind it changes.

## Calling it

The endpoint speaks the OpenAI API, so any client already pointed at OpenAI
works by changing the base URL and key:

```python
client = OpenAI(base_url="https://<host>/<endpoint>/v1", api_key=key)
```

Models **corerun runs** also answer Anthropic's API on the same address, because
the engine serves both:

```python
client = Anthropic(base_url="https://<host>/<endpoint>", api_key=key)
```

Note the Anthropic base URL has no `/v1` — that SDK appends it. An endpoint that
only fronts external upstreams speaks whatever that provider speaks, so this
applies to deployments rather than to everything published.

## When a call fails

**401** — the key is wrong, or it belongs to a different endpoint. A key
identifies the endpoint, and the name in the URL has to agree with it.

**404 naming another address** — the endpoint moved. The body carries the URL
to use; the old one is no longer served.

**The wrong model answers** — check `corerun endpoints models <name>`. A request
naming no model, or a name the endpoint does not publish, does not route where
the caller assumed.

**413** — the request is larger than the endpoint accepts. Multimodal requests
carry base64 images and reach this sooner than people expect.

## Removing things

```bash
corerun endpoints delete <name>
```

**Deleting an endpoint breaks every application using its address**, and the
name cannot be reissued as the same URL to a different set of models without
that surprise. Refused while a deployment is still running behind it. Confirm
with the human first, and prefer removing an upstream to deleting the address.
