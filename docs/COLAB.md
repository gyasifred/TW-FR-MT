# Colab Workflow

These notebooks are interactive workflows, not evidence that training has run.
Private Drive contents have not been inspected in this development session.
No GPU availability, training duration, translation quality, or successful run is promised.

## Get the Updated Code First

Commit and push the complete updated working tree to your GitHub repository before
using Colab. The original GitHub repository does not contain local changes until
they are pushed. Set `REPO_URL` to your credential-free HTTPS GitHub URL and,
optionally, `BRANCH` to the branch containing the package and notebooks.
Alternatively, upload the complete updated project to `/content/TW-FR-MT` before
running setup. Uploading just a notebook is insufficient.

Setup clones only when `/content/TW-FR-MT` does not exist. It never pulls, resets,
switches branches, or overwrites an existing checkout/upload. A requested branch
mismatch stops setup for manual resolution. Check the printed commit ID.
The notebooks install `pip -e '.[train,app]'` using the project's
`transformers==4.57.1` pin in `pyproject.toml`, not Transformers v5.
Restart the runtime after installation if previously imported libraries conflict,
then rerun setup. Do not put credentials in clone URLs or notebook cells.

## Inventory and Mount Drive

Run [00_drive_inventory.ipynb](../notebooks/00_drive_inventory.ipynb) first.
It uses standard Colab Google authorization, `google.auth.default()` and Drive v3
`files.get` / paginated `files.list` to report metadata only for these folders:

| Role | Exact ID | Link |
| --- | --- | --- |
| Training data (`DATA_FOLDER_ID`) | `139FuEvqxcZzVglh0qNBIIDaXusjpgFuh` | [Data folder](https://drive.google.com/drive/folders/139FuEvqxcZzVglh0qNBIIDaXusjpgFuh?usp=drive_link) |
| Model outputs (`MODEL_FOLDER_ID`) | `13irIvPsqnryP_NJ5y6PneFKrQDJs5Qm6` | [Model folder](https://drive.google.com/drive/folders/13irIvPsqnryP_NJ5y6PneFKrQDJs5Qm6?usp=drive_link) |

The inventory does not read file contents or download weights. It lists direct
children, including names, IDs and shortcut metadata. Inspect nested folders
explicitly if necessary. A failed request needs account-access/API checks; there
is no public download fallback. Names and IDs may themselves be sensitive: clear
inventory outputs before sharing. See the official
[files.get](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/get)
and [files.list](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/list)
references for these metadata operations.

Each notebook mounts Drive explicitly:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The inventory notebook uses the paths recorded in your saved Colab inventory:
`DATA_ROOT = Path('/content/drive/MyDrive/data')` and
`DRIVE_ROOT = Path('/content/drive/MyDrive/MT')`.
Its selected checkpoint is `MT/OPUS-mt-tw-fr-tuned`. It checks the exact
translation filenames: `train_en`, `train_twi`, `train_fr`; `val_en`, `val_tw`,
`val_fr`; and `test_en`, `test_twi`, `test_fr`. These checks verify presence,
not sentence alignment or checkpoint compatibility. Other notebooks have their
own configuration cells; running the inventory does not configure them.
`MODEL_ROOT` is the corresponding inference root. No `gdown`, public Drive links,
or embedded credentials are used. Mounting lets notebook code access your Drive;
review code before granting access. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

### Path Inventory

| Maintained file | Configured storage |
| --- | --- |
| `00_drive_inventory.ipynb` | All nine recorded translation files under `MyDrive/data`; checkpoint under `MyDrive/MT` |
| `01_colab_train.ipynb` | Same roots; English/Twi supplied splits; outputs under `MT/runs/en-tw-run-001` |
| `02_colab_translate.ipynb` | `MT/OPUS-mt-tw-fr-tuned` |
| `configs/train.example.json` | Validated splits under `MT/prepared_data/en-tw-run-001`; same training output directory |
| `configs/models.drive.json` | `OPUS-mt-tw-fr-tuned`, relative to the mounted `MT` root |
| `app.py` / `twfr-mt serve` | Drive registry and `/content/drive/MyDrive/MT` by default; CLI/environment overrides remain supported |

The `/content/TW-FR-MT` checkout and `/content/twfr-inputs` and
`/content/twfr-splits` workspaces are local Colab paths, not Drive folders.
Container `/models` paths remain deployment mount points. Archived `legacy/`
files and `configs/models.example.json` are historical/schema examples, not
configured Colab entry points. No checkpoint is assumed for an unverified
reverse or pivot translation route.

## Train from Your Drive Data

Open [01_colab_train.ipynb](../notebooks/01_colab_train.ipynb). It uses
`/content/drive/MyDrive/data/train_en` and `train_twi`, `val_en` and `val_tw`,
and `test_en` and `test_twi` for English-to-Twi training. French filenames
are also configured for experiments with a compatible model. Verify sentence
alignment and language direction before training. Existence checks and strict data parsing stop on missing,
blank, empty, or unequal inputs. Metadata alone cannot establish alignment.

The primary model is configurable Marian
[`Helsinki-NLP/opus-mt-en-tw`](https://huggingface.co/Helsinki-NLP/opus-mt-en-tw),
whose publisher model card specifies **English to Twi**. Select English source
and Twi target data. It is not a supported Twi/French model merely because labels
are changed. Other pairs require a compatible model and data. Use real tokenizer
language codes/prefixes only where that model requires them; never invent `tw` or
`fr` special tokens. The default Marian path uses null tokenizer language codes
and an empty source prefix.

Selected Drive inputs are copied to `/content/twfr-inputs/en-tw-run-001` to
reduce Drive I/O. The notebook preserves supplied train/validation/test membership
and validates all three splits without resplitting. It writes `manifest.json` plus:

```text
train.source.txt       train.target.txt
validation.source.txt  validation.target.txt
test.source.txt        test.target.txt
```

Parsing checks row counts and nonblank text; the audit rejects exact duplicate
pairs and cross-split normalized-source overlap. Neither check proves semantic
alignment. No rows are silently removed or reassigned. Input hashes and split
counts are saved to `/content/drive/MyDrive/MT/prepared_data/en-tw-run-001`
without replacing different saved data. Local corpus copies and validated files
contain private text. Test predictions or scores are not used for development.

The optional, secondary bundled example uses aligned
`TW_FR_EN_corpus/data/total/total_en.txt` and `total_tw.txt`, each with 10,708
lines. Use the CLI `twfr-mt prepare` to create new grouped splits from total
files when deliberately running that separate experiment.
Never use legacy split files for new training: shipped `test_fr.txt` has 1,070
lines versus 1,071 for English/Twi. The legacy audit found normalized source-key
overlaps of 45 train/validation, 40 train/test and 2 validation/test. These are
archival inputs; originals remain unchanged. Do not silently truncate, pad,
repair, or combine their test data. Rebuild from aligned total files instead.

`TrainingConfig` receives only train/validation paths, plus model/language,
tokenizer, optimization, output and resume settings. Start with
`run_training(config)`. Use validation for development. Freeze the model and
decoding settings before generating final held-out test predictions. Do not tune
on test scores. The CLI equivalent uses your own reviewed configuration:

```bash
twfr-mt train --config configs/train.example.json
```

## Persistence and Resume

Choose a unique `RUN_NAME`. Training writes directly to
`DRIVE_ROOT/runs/<RUN_NAME>`, with full `checkpoint-<step>` directories and a
`final` model/tokenizer export on completion. Inspect saved files rather than
assuming interrupted writes are complete. Preserve `run_config.json`,
`training_summary.json`, and prepared-data manifests for provenance.

To resume, use a fresh runtime, select the same paths, model revision, data and
settings, and set `RESUME_FROM_CHECKPOINT` to the explicit full
`RUN_DIR/checkpoint-<step>` path. Full Trainer state includes optimizer, scheduler,
RNG and trainer state, not just model weights. The runner checks provenance and
rejects a changed dataset/configuration, an incomplete checkpoint, a nonempty
fresh run, or an existing `final` export. Resolve any stale lock only after
confirming no training process remains. A completed final export is for inference,
not full-state resume. If rerunning preparation in the same runtime, inspect
existing local directories before removing them manually; the notebook will not
overwrite them.

Colab runtimes are temporary, resources vary, and idle/runtime limits apply.
Drive I/O can be slow, interrupted or quota-limited; local input copies reduce
small reads but checkpoint writes still use Drive. Keep adequate Drive space and
verify important artifacts before ending a session. These constraints are
documented in the [official FAQ](https://research.google.com/colaboratory/faq.html).

## Translate Interactively

Open [02_colab_translate.ipynb](../notebooks/02_colab_translate.ipynb), mount the
authorized model folder, and set an actual `CHECKPOINT_RELATIVE` under
`MODEL_ROOT`. The configured model root is `/content/drive/MyDrive/MT`, with
`OPUS-mt-tw-fr-tuned` selected for Twi-to-French translation. The notebook writes
its route registry under `/content/twfr-registries` after configuration.
It refuses to overwrite a different registry and never copies or
uploads weights. `create_app(registry_path, model_root, device='auto')` returns a
Gradio Blocks application. Both `LAUNCH_UI` and `ENABLE_PUBLIC_SHARE` default to
false. `app.launch(share=ENABLE_PUBLIC_SHARE)` runs only after explicit opt-in;
true sharing creates a public link. Keep sensitive text out of a shared app.

The selected folder must contain complete, compatible model and tokenizer
assets. The notebook explains separate conversion of trusted older `.bin`
checkpoints to safetensors without modifying the originals. For other routes,
select the actual compatible checkpoints and languages. For pivots, adjacent
step languages must match; folder names alone do not verify model support.

Colab is not 24/7 hosting. Free managed runtimes without positive compute units
restrict using a web UI as the primary interface and may terminate such sessions.
Do not bypass restrictions or use keepalive workarounds. Review the
[Colab FAQ](https://research.google.com/colaboratory/faq.html) and use the
[deployment workflow](DEPLOYMENT.md) for a hosted application.
