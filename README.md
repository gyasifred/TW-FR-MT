# Twi-French Machine Translation

## Overview

Train sequence-to-sequence translation models in Google Colab, retain
checkpoints on Google Drive, and serve direct or English-pivot translation
through a Gradio conversation interface. Each message is translated
independently: this is machine translation, not an instruction-following chatbot.

The maintained implementation is `twfr_mt/`. Original research scripts and
their historical tutorial are retained under [legacy/](legacy/README.md).
Original corpus files are unchanged.

## Features

- Strict UTF-8 aligned-corpus validation with accent/case preservation.
- Reproducible source-grouped splits, deduplication and hashed data provenance.
- Seq2SeqTrainer training with dynamic padding, GPU-aware precision and full-state resume.
- Local safetensors checkpoint loading, bounded model caching and direct/pivot routes.
- Case-sensitive SacreBLEU, chrF and TER with metric signatures.
- Local-first Gradio serving; authentication or explicit consent for remote sharing.
- A non-root CPU Docker image and deployment instructions.

## Quickstart Guide

### Colab and Google Drive

Follow the notebooks in order:

1. [Drive inventory](notebooks/00_drive_inventory.ipynb): authenticate and identify actual files.
2. [Train and resume](notebooks/01_colab_train.ipynb): validate aligned data, prepare splits, train and export.
3. [Load and translate](notebooks/02_colab_translate.ipynb): configure existing checkpoints and launch the UI.

Provided Drive folders:

- [Training data](https://drive.google.com/drive/folders/139FuEvqxcZzVglh0qNBIIDaXusjpgFuh)
- [Model checkpoints](https://drive.google.com/drive/folders/13irIvPsqnryP_NJ5y6PneFKrQDJs5Qm6)

Folder IDs do not establish mounted paths, filenames, model formats or language
directions. Confirm those with authenticated inventory. These private folders
have not been inspected or downloaded by the local application.

See [Colab setup](docs/COLAB.md). A GitHub clone in Colab only contains changes
already pushed to GitHub; alternatively upload the source bundle produced by
`python scripts/package_colab.py` and extract it under `/content`.

### Local Serving

Use Python 3.10 or newer; the Docker image uses Python 3.11.

```bash
python -m pip install -e ".[inference,app]"
twfr-mt serve --registry configs/models.example.json --model-root /path/to/checkpoints
```

Edit the registry to match actual checkpoint directories below your model root.
The example directory names are placeholders, not supplied trained models.
No checkpoint is downloaded or loaded until translation is requested. Models
must be encoder-decoder Hugging Face exports with their tokenizer files.

For a trusted older `.bin` export, convert it explicitly:

```bash
twfr-mt export-checkpoint --source /trusted/old-model --destination /models/tw-fr/final --allow-legacy-bin
```

The server defaults to localhost with sharing disabled. For authenticated
remote serving, set `TWFR_USERNAME` and `TWFR_PASSWORD` in the environment;
never commit credentials. Text is not intentionally persisted in chat history
or sent to third-party translation APIs. Review hosting/proxy logs separately.

### Training and Evaluation

```bash
python -m pip install -e ".[train]"
twfr-mt prepare --source source.txt --target target.txt --output-dir prepared/pair
twfr-mt train --config configs/train.example.json
twfr-mt evaluate --predictions predictions.txt --references references.txt
```

Adapt the training config's paths, model and language pair before execution.
The example English-to-Twi base model does not establish compatibility with
French-to-Twi or Twi-to-French fine-tuning. Multilingual models may require
explicit source/target tokenizer language tokens or source prefixes.

The original bundled French test file has 1,070 rows, while the Twi and English
test files have 1,071. Original source text also overlaps across splits.
Do not silently truncate, repair, or report clean held-out results from those
files. Use independently validated Drive data or rebuild grouped pairwise
splits from aligned total files. Pairwise splits are not automatically aligned
across independently prepared three-language experiments.

### Deployment and Tests

See [deployment options](docs/DEPLOYMENT.md). A hosted container cannot mount
your personal Colab Drive automatically; stage authorized checkpoint exports
onto a persistent model volume. Colab sessions and share links are not
permanent application hosting.

```bash
python -m pip install -e ".[dev,eval]"
python -m pytest -q
```

Lightweight tests do not require GPU packages. Tests with real ML dependencies
must be distinguished from dependency-injected orchestration tests. Training
success, translation quality, private Drive access and hosted deployment are
not established by a passing UI or unit-test suite alone.

See the [local validation record](docs/VALIDATION.md) for completed checks and
remaining environment-specific verification.

## Citation

If you find this project helpful, consider citing it:

```bibtex
@Article{bdcc7020114,
    AUTHOR = {Gyasi, Frederick and Schlippe, Tim},
    TITLE = {Twi Machine Translation},
    JOURNAL = {Big Data and Cognitive Computing},
    VOLUME = {7},
    YEAR = {2023},
    NUMBER = {2},
    ARTICLE-NUMBER = {114},
    URL = {https://www.mdpi.com/2504-2289/7/2/114},
    ISSN = {2504-2289},
    DOI = {10.3390/bdcc7020114}
}
```

## License
This repository is open-source and licensed under the [MIT license](https://github.com/gyasifred/TW-FR-MT/blob/main/LICENSE).

## About the Author
Maintained by [Frederick Gyasi](https://github.com/gyasifred), passionate about language technologies and cross-cultural communication.

