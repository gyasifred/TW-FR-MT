# Deployment

The inference service runs independently of the training environment. A
deployment consists of a compatible model export, a language-route registry,
the application runtime, and appropriate network and access controls.

## Model Artifacts

The configured Twi-to-French checkpoint is
`/content/drive/MyDrive/MT/OPUS-mt-tw-fr-tuned`. A deployable export includes
model configuration, safetensors weights, tokenizer assets, and any generation
configuration needed for consistent decoding. Training-state checkpoints and
completed inference exports have different purposes.

Before transferring artifacts, verify their language direction, file integrity,
license, and authorization for the intended hosting environment. Transfer only
the selected model artifacts; training corpora are not required for inference.
Compare source and destination file hashes and retain the source revision,
model revision, and route configuration with the deployment record.

Docker containers and hosted services do not inherit Colab's mounted Drive or
Google browser authorization. Provision artifacts through an authorized transfer
or authenticated artifact store, then expose the model directory to the runtime
as a read-only volume. Credentials belong in runtime secret storage, not source
files, notebook outputs, image layers, or public download URLs.

## Route Configuration

`configs/models.drive.json` selects `OPUS-mt-tw-fr-tuned` for Twi-to-French
translation. Checkpoint paths in the registry are relative to the model root:

```text
/models/
  OPUS-mt-tw-fr-tuned/
    config.json
    model.safetensors
    ... tokenizer and generation assets
```

Model weights may use safetensors shards with an index rather than one weight
file. The tokenizer assets must match the model export.

For an English-to-Twi export produced by the training notebook, the relative
path is `runs/en-tw-run-001/final` when the original Drive directory structure
is retained. Configure a corresponding English-to-Twi route before serving it.
Additional direct or English-pivot routes require a compatible checkpoint for
every step. Adjacent steps must use consistent intermediate languages;
tokenizer language codes and prefixes must match the models' training settings.

## Application Runtime

Install inference and interface dependencies:

```bash
python -m pip install -e '.[inference,app]'
twfr-mt serve --registry configs/models.drive.json --model-root /models
```

This command assumes the staged directory is mounted at `/models`. In Colab,
the corresponding root is `/content/drive/MyDrive/MT`. The default command
`python app.py` uses the Drive registry and root; hosted environments should
set the model root explicitly or through `TWFR_MODEL_ROOT`.

The service binds to localhost by default. For remote binding, provision
`TWFR_USERNAME` and `TWFR_PASSWORD` or explicitly authorize public access with
`--allow-public`. Use HTTPS and infrastructure-level access controls for
network exposure. Public Gradio share links are not required for hosted serving.

| Setting | Purpose |
| --- | --- |
| `TWFR_REGISTRY` | Route-registry path |
| `TWFR_MODEL_ROOT` | Root directory containing model exports |
| `TWFR_USERNAME`, `TWFR_PASSWORD` | Application authentication credentials |
| `GRADIO_SERVER_NAME` | Listening interface |
| `PORT` | Listening port; default 7860 |

## Docker

The [Dockerfile](../Dockerfile) defines a non-root, CPU-oriented Python 3.11
runtime with model storage mounted at `/models`. Keep private artifacts outside
the image build context.

For PowerShell, provision `TWFR_MODEL_STAGE` as the absolute host directory
containing `OPUS-mt-tw-fr-tuned`, together with the application credentials in
the host environment:

```powershell
if (-not $env:TWFR_MODEL_STAGE) { throw 'TWFR_MODEL_STAGE must identify the model staging directory.' }
if (-not $env:TWFR_USERNAME -or -not $env:TWFR_PASSWORD) { throw 'Application credentials are required.' }
$ModelStage = (Resolve-Path -LiteralPath $env:TWFR_MODEL_STAGE).Path
$Registry = (Resolve-Path 'configs/models.drive.json').Path

docker build -f Dockerfile -t twfr-mt .
docker run --rm -p 127.0.0.1:7860:7860 `
  --env TWFR_USERNAME --env TWFR_PASSWORD `
  --mount "type=bind,source=$ModelStage,target=/models,readonly" `
  --mount "type=bind,source=$Registry,target=/runtime/models.json,readonly" `
  twfr-mt twfr-mt serve --registry /runtime/models.json --model-root /models --host 0.0.0.0
```

The host port remains bound to localhost. A separately configured HTTPS reverse
proxy can provide controlled remote access. Read-only model mounts limit writes
to artifacts; the container user still requires read permissions. GPU serving
requires a compatible container runtime and hardware configuration beyond the
CPU image.

## Hosted Containers

A Docker-based hosting service must provide model artifacts before application
startup, expose the application port, and supply authentication through runtime
secrets. A Hugging Face Docker Space uses `sdk: docker` and `app_port: 7860`
in its README metadata when serving on that port. The application must bind to
`0.0.0.0` inside the container.

Model publication and application hosting are separate authorization decisions.
Avoid placing restricted artifacts in public repositories or image layers.
Treat runtime storage as temporary unless persistent storage is explicitly
provisioned. Colab sessions and share links are not substitutes for persistent
application hosting.

## Acceptance Criteria

Before enabling access, verify that every registered route loads its intended
model and tokenizer, produces output for a non-sensitive test sentence, and
enforces the configured input and generation limits. Test authentication,
artifact permissions, log handling, restart behavior, and missing-artifact
failures. Runtime validation does not establish translation quality.

For final research evaluation, fix the model and decoding settings and generate
one prediction per held-out source row in order. Evaluate against that
partition's references and retain the predictions, metric signatures, corpus
manifest, and model configuration. Do not use final test scores to select
checkpoints or tune parameters.

## Platform References

- [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
- [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Google Drive file downloads](https://developers.google.com/workspace/drive/api/guides/manage-downloads)
- [Colab runtime limits](https://research.google.com/colaboratory/faq.html)
