# Deployment

Train interactively in Colab, then deploy a verified model export to an actual
hosting runtime. A Colab share link is not durable hosting. This document does
not claim that training, Docker, or a Space has been executed successfully here.

## Export Authorized Artifacts First

The supplied private folders have separate roles:

- Data: [`139FuEvqxcZzVglh0qNBIIDaXusjpgFuh`](https://drive.google.com/drive/folders/139FuEvqxcZzVglh0qNBIIDaXusjpgFuh?usp=drive_link).
- Models: [`13irIvPsqnryP_NJ5y6PneFKrQDJs5Qm6`](https://drive.google.com/drive/folders/13irIvPsqnryP_NJ5y6PneFKrQDJs5Qm6?usp=drive_link).

Use [the inventory notebook](../notebooks/00_drive_inventory.ipynb) and the
[Colab instructions](COLAB.md) to authorize access, add My Drive shortcuts if
necessary, and select actual paths. Folder contents/names were not accessible to
this development session. No filesystem name is inferred from either ID.

Docker and Hugging Face Spaces cannot use Colab's `drive.mount()` magic or inherit
your browser's Google authorization. Stage artifacts before starting deployment:

1. In your authenticated Drive/Colab session, identify a complete authorized
   `final` export. Verify `config.json`, model weights, tokenizer assets and
   generation configuration. Review the license and your rights to use and
   redistribute the model and any training-derived artifacts.
2. Explicitly export/download only the checkpoint(s) you authorize to a private
   host staging directory, preserving filenames. Use authenticated Drive export
   or a manually downloaded archive from the mounted checkpoint; inspect archive
   contents and extraction paths before extracting. Do not download the whole
   data folder or expose it through the app. Compare SHA-256 hashes between the
   source and staged files. Keep an access-controlled provenance record.
3. Arrange staged directories to match your registry's relative paths. Mount the
   staging root read-only as `/models` or provision the same tree through a
   private authenticated artifact store before app startup. Do not move/delete
   the source Drive files. The notebooks do not upload checkpoints to a hosting
   provider.
4. Keep credentials in runtime secret storage, never source, image layers, public
   URLs, notebook output, or Docker build arguments. If using a private artifact
   store, authorize its upload/download explicitly and fail startup when an
   artifact is missing. Do not fall back to a different model silently.

Do not make Drive folders public, use `gdown` or unauthenticated public-folder
downloads, or bake Google credentials into an image to bridge the environments.
For programmatic authenticated export, implement the Drive API prepare step
separately with least-privilege credentials; review the official
[Drive download guide](https://developers.google.com/workspace/drive/api/guides/manage-downloads).

## Registry and Runtime

Use the package's route schema. Paths are relative to `--model-root`, never a
Colab `/content/drive/...` path. This English-to-Twi example assumes your authorized
staging tree actually contains `/models/en-tw/final`:

```json
{
  "routes": [
    {
      "id": "en-tw",
      "label": "English to Twi",
      "source_lang": "en",
      "target_lang": "tw",
      "steps": [
        {
          "source_lang": "en",
          "target_lang": "tw",
          "checkpoint": "en-tw/final",
          "source_tokenizer_lang": null,
          "target_tokenizer_lang": null,
          "source_prefix": ""
        }
      ]
    }
  ]
}
```

If you preserve the notebook's `runs/en-tw-run-001/final` layout instead, use
that relative path in the registry. Edit and review
`configs/models.example.json` for the tree you actually staged; example registry
entries do not prove corresponding model weights exist.

For a direct Twi-to-French checkpoint, the route/step languages are `tw` and `fr`
with, for example, `checkpoint: "tw-fr/final"`. Supply that real compatible model
yourself. For pivots, configure ordered steps and a real checkpoint for each leg;
adjacent languages must match. Tokenizer codes and prefixes must come from those
models, not invented Twi/French tokens. The primary
[`Helsinki-NLP/opus-mt-en-tw` model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-tw)
supports English to Twi, not Twi/French.

After installing the project with its pinned dependencies:

```bash
pip install -e '.[train,app]'
twfr-mt serve --registry configs/models.example.json --model-root /models
```

The Python interface is `twfr_mt.ui.create_app(registry_path, model_root,
device='auto')`, returning Gradio Blocks. Public `share=True` is not needed for
normal hosting. Bind the host service to the container interface and expected
port using the deployment's launch configuration, then use the platform's HTTPS
endpoint and authentication. Restrict access, request sizes and concurrency;
avoid logging user text. Confirm all registered routes and model/tokenizer files
load before opening access. Test a known non-sensitive sentence for each route.

## Docker

The root `Dockerfile` builds a CPU image, exposes port 7860, and uses
`CMD ["twfr-mt", "serve", "--host", "0.0.0.0"]` (not an `ENTRYPOINT`).
Keep private model/data exports outside the build
context and out of image layers. A host bind mount is separate from Colab Drive
mounting. Before the following PowerShell commands, provision `TWFR_USERNAME`
and `TWFR_PASSWORD` in the host environment using your secret manager. Do not
commit them or put their literal values in commands or shared notebook outputs.
The CLI requires authentication for remote binding unless the operator explicitly
chooses `--allow-public`:

```powershell
docker build -f Dockerfile -t twfr-mt .
$ModelStage = (Resolve-Path 'C:/private/twfr-models').Path
$Registry = (Resolve-Path 'configs/models.example.json').Path
if (-not $env:TWFR_USERNAME -or -not $env:TWFR_PASSWORD) { throw 'Provision authentication through the host secret manager first.' }
docker run --rm -p 127.0.0.1:7860:7860 `
  --env TWFR_USERNAME --env TWFR_PASSWORD `
  --mount "type=bind,source=$ModelStage,target=/models,readonly" `
  --mount "type=bind,source=$Registry,target=/runtime/models.json,readonly" `
  twfr-mt twfr-mt serve --registry /runtime/models.json --model-root /models --host 0.0.0.0
```

Replace the staging directory with your authorized, verified host path. CPU is
the initial baseline; GPU use requires compatible hardware, container runtime,
and project dependencies, not a Colab runtime selection. Configure HTTPS and
access control before exposing beyond localhost. Read-only mounts limit writes
to the staged files; see [Docker's bind-mount documentation](https://docs.docker.com/engine/storage/bind-mounts/).

## Hugging Face Spaces

Use a Docker Space with a reviewed root `Dockerfile`, `sdk: docker`, and
`app_port: 7860` in the Space README metadata if the app listens on 7860. Review
the Dockerfile and launch command; the app must listen on `0.0.0.0` inside
the container. Provision authorized checkpoints from private staging/artifact
storage before startup; do not expect the Space to mount Google Drive. Do not
publish models in a public Space repository or image without explicit permission.

Set `TWFR_USERNAME` and `TWFR_PASSWORD` in Space Settings secrets for the CLI's
remote-bind authentication check. Deliberately public deployments instead require
explicit `--allow-public` in their launch command; do not use that flag to bypass
an intended access boundary. Set secrets in Space Settings, not committed files
or build arguments. Account
for the container user's filesystem permissions. Treat runtime disk as ephemeral
unless you configure the platform's storage options, and verify the current
hardware and persistence offering for your account. See the official
[Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker).
Do not run training during image builds or assume a build-time GPU is available.

## Final Evaluation

Freeze the model and decoding settings, generate one prediction per held-out
source row in order, and evaluate against that same prepared split's reference:

```bash
twfr-mt evaluate --predictions predictions.txt --references test.target.txt
```

Use your prepared `test.target.txt`, not the misaligned legacy test files.
Preserve the split manifest, predictions, metric signatures and model revision
privately for reproducibility. Never tune hyperparameters, routes, or checkpoint
selection on these final test scores.
