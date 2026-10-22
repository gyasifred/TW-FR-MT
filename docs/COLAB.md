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

Open both folders with your authorized account. Add a shortcut to My Drive if
needed. Use Colab's file browser to choose each actual mounted folder/shortcut
path. IDs and API names are not filesystem paths; do not infer folder names from
IDs. Set `DATA_ROOT` manually. The model-output configuration starts with
`DRIVE_ROOT = Path('/content/drive/MyDrive/TW-FR-MT')`; this is a configurable
example, not a claim that the supplied model folder has that name.
`MODEL_ROOT` is the corresponding inference root. No `gdown`, public Drive links,
or embedded credentials are used. Mounting lets notebook code access your Drive;
review code before granting access. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Train from Your Drive Data

Open [01_colab_train.ipynb](../notebooks/01_colab_train.ipynb). Choose actual
UTF-8 parallel-text filenames from the authenticated inventory. `source_en.txt`
and `target_tw.txt` are placeholders, not observed Drive filenames. Set
`DATA_SELECTION_CONFIRMED = True` only after verifying paths, sentence alignment,
and language direction. Existence checks and strict data parsing stop on missing,
blank, empty, or unequal inputs. Metadata alone cannot establish alignment.

The primary model is configurable Marian
[`Helsinki-NLP/opus-mt-en-tw`](https://huggingface.co/Helsinki-NLP/opus-mt-en-tw),
whose publisher model card specifies **English to Twi**. Select English source
and Twi target data. It is not a supported Twi/French model merely because labels
are changed. Other pairs require a compatible model and data. Use real tokenizer
language codes/prefixes only where that model requires them; never invent `tw` or
`fr` special tokens. The default Marian path uses null tokenizer language codes
and an empty source prefix.

Selected Drive inputs are copied to `/content/twfr-inputs/<RUN_NAME>` before
preparation to reduce Drive I/O. `prepare_splits(source, target, outputdir,
seed=42, validation_fraction=0.1, test_fraction=0.1)` returns a manifest dictionary
and writes `manifest.json` plus:

```text
train.source.txt       train.target.txt
validation.source.txt  validation.target.txt
test.source.txt        test.target.txt
```

Preparation normalizes text, deduplicates cleaned pairs and groups normalized
source texts before splitting. Fractions apply to source groups, so actual row
fractions can differ. The notebook validates source disjointness and alignment
without using test predictions or scores for decisions. It saves these prepared
splits and the manifest to `DRIVE_ROOT/prepared_data/<RUN_NAME>` without replacing
different saved data. Local corpus copies and prepared files contain private text.

The optional, secondary bundled example uses aligned
`TW_FR_EN_corpus/data/total/total_en.txt` and `total_tw.txt`, each with 10,708
lines. Explicitly select those paths in the notebook to use it instead of Drive.
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
`MODEL_ROOT`. The notebook writes an English-to-Twi registry to Drive after
configuration. It refuses to overwrite a different registry and never copies or
uploads weights. `create_app(registry_path, model_root, device='auto')` returns a
Gradio Blocks application. Both `LAUNCH_UI` and `ENABLE_PUBLIC_SHARE` default to
false. `app.launch(share=ENABLE_PUBLIC_SHARE)` runs only after explicit opt-in;
true sharing creates a public link. Keep sensitive text out of a shared app.

Twi/French routes require your actual compatible checkpoint(s); the schema-only
example does not supply them. For pivots, adjacent step languages must match.
Do not relabel the default English-to-Twi export.

Colab is not 24/7 hosting. Free managed runtimes without positive compute units
restrict using a web UI as the primary interface and may terminate such sessions.
Do not bypass restrictions or use keepalive workarounds. Review the
[Colab FAQ](https://research.google.com/colaboratory/faq.html) and use the
[deployment workflow](DEPLOYMENT.md) for a hosted application.
