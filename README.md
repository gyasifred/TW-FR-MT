# Twi-French Machine Translation

A research software framework for training, evaluating, and serving
sequence-to-sequence machine translation models. The framework supports direct
translation and two-stage translation through English, with checkpoints stored
on Google Drive and a Gradio interface for interactive inference. Each input
is translated independently; conversation history is not used as model context.

## Architecture

The `twfr_mt` package separates corpus preparation, model training, evaluation,
checkpoint loading, and application serving. Training uses Hugging Face
`Seq2SeqTrainer`. Inference loads local encoder-decoder model exports through
explicit language-pair routes. A bounded cache limits the number of models
retained by the translation service.

| Component | Responsibility |
| --- | --- |
| `twfr_mt/data.py` | Parallel-text parsing, partition validation, grouped splitting, and file provenance |
| `twfr_mt/training.py` | Fine-tuning, resumable training, and model export |
| `twfr_mt/inference.py` | Local checkpoint loading and direct or English-pivot translation |
| `twfr_mt/metrics.py` | SacreBLEU, chrF, and translation edit rate (TER) evaluation |
| `twfr_mt/ui.py` | Gradio translation interface |
| `twfr_mt/cli.py` | Command-line preparation, training, evaluation, and serving |

## Installation

Python 3.10 or later is required. Install the dependency group for the intended
workload:

```bash
python -m pip install -e ".[train,app]"
```

Inference-only environments can use `.[inference,app]`. The dependency
specification in [pyproject.toml](pyproject.toml) pins Transformers to version
4.57.1.

## Google Colab

The notebooks provide three entry points:

1. [Drive inventory](notebooks/00_drive_inventory.ipynb): authenticate and inspect available files.
2. [Training](notebooks/01_colab_train.ipynb): validate the supplied partitions, fine-tune, and export a model.
3. [Translation](notebooks/02_colab_translate.ipynb): load an existing checkpoint and construct the interface.

| Resource | Configured location |
| --- | --- |
| [Parallel corpora](https://drive.google.com/drive/folders/139FuEvqxcZzVglh0qNBIIDaXusjpgFuh) | `/content/drive/MyDrive/data` |
| [Model storage](https://drive.google.com/drive/folders/13irIvPsqnryP_NJ5y6PneFKrQDJs5Qm6) | `/content/drive/MyDrive/MT` |
| Twi-to-French inference checkpoint | `/content/drive/MyDrive/MT/OPUS-mt-tw-fr-tuned` |
| Training output | `/content/drive/MyDrive/MT/runs/en-tw-run-001` |

The training notebook defaults to English-to-Twi fine-tuning with
`Helsinki-NLP/opus-mt-en-tw`. The inference notebook selects the separate
Twi-to-French checkpoint. Each experiment requires a compatible base model,
tokenizer, source language, and target language.

See [Colab Training and Inference](docs/COLAB.md) for the complete workflow.

## Data and Experimental Design

Parallel corpora consist of UTF-8 source and target files with one aligned
sentence per line. Parsing preserves case and accents while normalizing Unicode
and whitespace. Empty inputs, blank rows, and unequal line counts are rejected.
Semantic alignment requires separate inspection.

The Colab workflow preserves the supplied training, validation, and test
partitions. It rejects duplicate sentence pairs and normalized-source overlap
across partitions. For experiments beginning with an unpartitioned corpus, the
preparation command groups equivalent source sentences before partitioning and
records input and output hashes in a manifest.

Validation data support model selection. The test partition is reserved for
evaluation after model and decoding settings are fixed. Independently prepared
language-pair partitions do not establish a shared three-language test set.

The bundled archival test files contain unequal language-specific line counts
and cross-partition source overlap. They require a separate corpus audit and
repartitioning from aligned source material before use in a new experiment.

## Training and Evaluation

After the training notebook saves the validated partitions, command-line
training is available through:

```bash
twfr-mt train --config configs/train.example.json
```

The configuration reads validated files from
`/content/drive/MyDrive/MT/prepared_data/en-tw-run-001`. A fresh experiment
requires an unused output directory. Resuming an interrupted run requires a
complete Trainer checkpoint and compatible data and configuration.

Evaluate one prediction per test reference, in the same order:

```bash
twfr-mt evaluate --predictions predictions.txt --references /content/drive/MyDrive/MT/prepared_data/en-tw-run-001/test.target.txt
```

Evaluation reports case-sensitive SacreBLEU, chrF, and TER with metric
signatures. Interpret these corpus-level measures alongside linguistic error
analysis appropriate to the research question.

## Interactive Translation

With Drive mounted in Colab:

```bash
twfr-mt serve --registry configs/models.drive.json --model-root /content/drive/MyDrive/MT
```

The registry selects `OPUS-mt-tw-fr-tuned` for Twi-to-French translation.
Checkpoints must include compatible model configuration, weights, and tokenizer
assets. Loading is deferred until translation is requested. Additional direct
or pivot routes require an explicit checkpoint for every translation step.

Serving defaults to localhost with public sharing disabled. Remote access
requires authentication through `TWFR_USERNAME` and `TWFR_PASSWORD`, or explicit
authorization for public access. For deployment outside Colab, stage model
exports on the hosting system and specify its model root. See
[Deployment](docs/DEPLOYMENT.md) for artifact preparation and access control.

## Software Verification

```bash
python -m pip install -e ".[dev,eval]"
python -m pytest -q
python -m ruff check twfr_mt tests scripts
```

Tests cover corpus validation, routing, training configuration, checkpoint
handling, and serving controls. An optional real-dependency integration test
exercises training and inference with a small synthetic model. Software tests
do not establish translation performance on the research corpus. See
[Validation and Reproducibility](docs/VALIDATION.md) for verification scope.

## Research Reference

Gyasi F, Schlippe T. Twi Machine Translation. *Big Data and Cognitive Computing*.
2023;7(2):114. [doi:10.3390/bdcc7020114](https://doi.org/10.3390/bdcc7020114).

```bibtex
@article{gyasi2023twi,
  author = {Gyasi, Frederick and Schlippe, Tim},
  title = {Twi Machine Translation},
  journal = {Big Data and Cognitive Computing},
  year = {2023},
  volume = {7},
  number = {2},
  pages = {114},
  doi = {10.3390/bdcc7020114}
}
```

The [research archive](legacy/README.md) contains the earlier experimental
scripts and tutorial. The maintained package is documented separately from
those archival materials.

## License

The source code is distributed under the [MIT License](LICENSE). Dataset and
pretrained-model licenses govern their respective artifacts.
