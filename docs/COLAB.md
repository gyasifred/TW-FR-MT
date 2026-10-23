# Colab Training and Inference

The Colab notebooks support authenticated corpus access, reproducible
fine-tuning, and interactive translation. Google Drive provides persistent
storage; local Colab directories provide temporary working space.

## Environment

Open the notebooks from the [repository](https://github.com/gyasifred/TW-FR-MT)
and execute cells in order. Training benefits from a GPU runtime. Availability,
memory capacity, and session duration depend on the account and runtime.

Notebook setup uses `/content/TW-FR-MT` as the source checkout and installs
`.[train,app]`. Existing checkouts are retained rather than overwritten or
automatically updated. The displayed commit identifies the source revision for
the session. Restart the runtime after installation if imported libraries
conflict with the project's Transformers 4.57.1 dependency.

Authentication uses standard Google authorization and Drive mounting:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Credentials belong in the authorization flow, not notebook cells or repository
URLs. Notebook outputs can contain sensitive file metadata and should be
cleared before public distribution.

## Storage Configuration

| Resource | Mounted location |
| --- | --- |
| [Training corpora](https://drive.google.com/drive/folders/139FuEvqxcZzVglh0qNBIIDaXusjpgFuh) | `/content/drive/MyDrive/data` |
| [Model artifacts](https://drive.google.com/drive/folders/13irIvPsqnryP_NJ5y6PneFKrQDJs5Qm6) | `/content/drive/MyDrive/MT` |
| Inference checkpoint | `/content/drive/MyDrive/MT/OPUS-mt-tw-fr-tuned` |
| Validated training inputs | `/content/drive/MyDrive/MT/prepared_data/en-tw-run-001` |
| Training output | `/content/drive/MyDrive/MT/runs/en-tw-run-001` |

The inventory notebook lists folder metadata and direct children without
reading corpus contents or loading model weights. It also checks the configured
file and checkpoint paths. Each notebook contains its own configuration cell.

### Corpus Files

These filenames are relative to `/content/drive/MyDrive/data`:

| Partition | English | Twi | French |
| --- | --- | --- | --- |
| Training | `train_en` | `train_twi` | `train_fr` |
| Validation | `val_en` | `val_tw` | `val_fr` |
| Test | `test_en` | `test_twi` | `test_fr` |

The Twi validation filename is `val_tw`; its training and test filenames use
`twi`. File presence establishes neither sentence alignment nor language
compatibility.

## Training

[01_colab_train.ipynb](../notebooks/01_colab_train.ipynb) defaults to English-to-Twi
fine-tuning with `Helsinki-NLP/opus-mt-en-tw`. It selects the English and Twi
columns from the corpus table. A different translation direction requires
compatible data, model, tokenizer settings, and any model-specific prefixes.

### Input Validation

The notebook copies the selected files to
`/content/twfr-inputs/en-tw-run-001`, verifies the copies against input hashes,
and preserves the supplied partition membership. It rejects empty files, blank
rows, unequal parallel lengths, duplicate exact pairs, and cross-partition
normalized-source overlap. Semantic alignment requires inspection beyond these
structural checks.

Validated files are stored under `/content/twfr-splits/en-tw-run-001`
and persisted to the configured Drive preparation directory:

```text
train.source.txt       train.target.txt
validation.source.txt  validation.target.txt
test.source.txt        test.target.txt
manifest.json
```

The manifest records language direction, input paths, hashes, partition counts,
and the partition-preservation policy. Existing saved inputs must match before
reuse. No rows are silently removed or reassigned.

### Model Fitting

`TrainingConfig` specifies the base model, language pair, tokenizer settings,
training and validation files, optimization parameters, and output directory.
The notebook calls `run_training(config)` to start fine-tuning. Alternatively,
after input preparation, run:

```bash
twfr-mt train --config configs/train.example.json
```

Use one training entry point for a fresh output directory. Validation data
support checkpoint and hyperparameter selection. Test data are excluded from
Trainer inputs and reserved for final evaluation.

### Checkpoints and Resume

The run directory contains full `checkpoint-<step>` directories, a completed
`final` export, `run_config.json`, and `training_summary.json`. Input manifests
are stored separately under `MT/prepared_data/en-tw-run-001`.

To resume an interrupted experiment, select its original configuration and set
`RESUME_FROM_CHECKPOINT` to a complete checkpoint directory within the run.
Resume checks require model weights, optimizer, scheduler, random-number and
Trainer state, together with compatible data and configuration provenance.
Some training-duration and runtime settings can be changed where permitted by
the runner. A completed `final` export is an inference artifact, not a
full-state resume checkpoint.

An existing final export or a nonempty fresh-run directory is not overwritten.
Before resolving a stale `.training.lock`, establish that no associated
training process is active. A fresh Colab runtime avoids collisions with
existing temporary input directories.

### Alternative Corpus Preparation

For an unpartitioned aligned corpus, `twfr-mt prepare` creates source-grouped
partitions with deterministic seeded allocation, pair deduplication, and hashed
provenance. This is separate from the notebook's partition-preservation workflow.
The bundled archival splits require auditing before reuse because they contain
unequal language-specific test counts and cross-partition source overlap.

## Interactive Inference

[02_colab_translate.ipynb](../notebooks/02_colab_translate.ipynb) selects
`/content/drive/MyDrive/MT/OPUS-mt-tw-fr-tuned` for Twi-to-French translation.
The checkpoint must contain compatible configuration, tokenizer, and weight
files. Its directory name alone does not establish the trained language pair.

The notebook creates a route registry under `/content/twfr-registries` and
constructs a Gradio application. `LAUNCH_UI` and `ENABLE_PUBLIC_SHARE` default
to `False`; application launch and public sharing require explicit selection.
The application translates each message independently.

Trusted legacy `pytorch_model.bin` checkpoints require explicit conversion to
a separate safetensors export before use with the default inference loader:

```python
from twfr_mt.export import export_checkpoint

export_checkpoint(
    '/content/drive/MyDrive/MT/OPUS-mt-tw-fr-tuned',
    '/content/drive/MyDrive/MT/OPUS-mt-tw-fr-tuned-safe',
    allow_legacy_bin=True,
)
```

Select the converted directory in `CHECKPOINT_RELATIVE` after conversion.
The source checkpoint is retained and an existing destination is not
overwritten. Only convert trusted artifacts.

## Operational Limits

Colab working directories are temporary. Drive persistence does not eliminate
interrupted writes, storage limits, or incomplete checkpoint risks; verify
artifacts before ending a session. Local input copies reduce repeated Drive
reads, while checkpoint writes still depend on Drive availability.

Public interface links expose a running application and are unsuitable for
sensitive text without appropriate access controls. Colab is not a persistent
hosting service. Consult the [official Colab FAQ](https://research.google.com/colaboratory/faq.html)
for runtime and web-interface restrictions, and use the
[deployment guide](DEPLOYMENT.md) for hosted inference.

## Configuration Reference

| Entry point | Configuration |
| --- | --- |
| Inventory notebook | All nine corpus files and the selected checkpoint |
| Training notebook | English/Twi inputs, validated partitions, and run directory |
| Inference notebook | Twi-to-French checkpoint and temporary route registry |
| `configs/train.example.json` | Persisted validated English/Twi files and training output |
| `configs/models.drive.json` | `OPUS-mt-tw-fr-tuned`, relative to the `MT` model root |
| `app.py` / `twfr-mt serve` | Drive registry and model root; command-line and environment overrides supported |

Container paths such as `/models` are deployment mount points, not Colab Drive
paths. `configs/models.example.json` illustrates additional route structures.
