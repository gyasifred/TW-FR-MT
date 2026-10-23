# Validation and Reproducibility

Software verification establishes the behavior of the implementation under
specified test conditions. Research evaluation establishes translation
performance on an identified corpus. Deployment acceptance establishes runtime
readiness in a particular environment. These forms of evidence are related
but not interchangeable.

## Test Coverage

| Area | Verification scope |
| --- | --- |
| Corpus parsing | UTF-8 handling, Unicode normalization, blank rows, empty files, and unequal parallel lengths |
| Partition integrity | Duplicate pairs, normalized-source overlap, deterministic grouping, and file provenance |
| Notebook configuration | Drive roots, exact filenames, partition preservation, checkpoint paths, and Python syntax |
| Training configuration | Parameter validation, output protection, data compatibility, and resume provenance |
| Inference | Route continuity, checkpoint containment, model caching, tokenizer settings, and generation limits |
| Application serving | Authentication requirements, explicit public access, and environment overrides |
| Metrics | Reference/prediction validation and metric computation |
| Model integration | Offline synthetic training, interruption, resume, export, and local inference |

Dependency-injected tests isolate orchestration and error handling. The
real-dependency integration test constructs a small random BART model and
synthetic tokenizer locally. It exercises Trainer checkpoint recovery and
safetensors export without downloading a pretrained model or accessing Drive.
Its synthetic outputs test pipeline execution, not learned translation quality.
The integration test is skipped when required machine-learning dependencies
are unavailable.

## Verification Commands

For lightweight checks:

```bash
python -m pip install -e '.[dev,eval]'
python -m pytest -q
python -m ruff check twfr_mt tests scripts
```

For real-dependency training integration, also install `.[train]`. Record the
source commit, Python version, dependency versions, test results, warnings,
and skipped tests with each verification run. A passing suite does not replace
execution in the intended Colab or deployment environment.

## Corpus and Model Validation

Corpus acceptance requires confirmed language direction, sentence alignment,
partition integrity, and appropriate usage rights. Matching row counts provide
a structural check, not proof that sentence pairs are translations.
Cross-partition source checks reduce exact-source leakage but do not detect
all paraphrases, near duplicates, or other forms of dependence.

Checkpoint acceptance requires complete model and tokenizer assets, compatible
language settings, and a documented relationship to the training configuration.
A model directory name does not establish language support. Use representative,
non-sensitive examples to check loading and decoding before corpus evaluation.

## Experimental Reproducibility

Retain the following artifacts for each experiment:

- Source revision, dependency versions, hardware, and random seeds.
- Corpus identifiers, eligibility rules, partition policy, and input hashes.
- Training configuration, model revision, tokenizer settings, and checkpoints.
- Validation-based selection criteria and fixed final decoding settings.
- Test predictions, aligned references, metric signatures, and error analysis.

The training notebook's manifest identifies input files and hashes while
preserving the supplied partitions. The separate corpus-preparation utility
also records its grouping, deduplication, and allocation policy. Training
provenance and checkpoint state support reproducibility but do not guarantee
bitwise agreement across hardware and software environments.

## Interpretation of Results

SacreBLEU, chrF, and translation edit rate summarize different aspects of
prediction-reference agreement. Comparisons require the same test population,
reference set, text-processing policy, and decoding conditions. Pivot and direct
routes should be evaluated on matched inputs and references.

Claims about practical translation quality require evidence beyond successful
execution or synthetic tests. Linguistic review should examine meaning
preservation, omissions, additions, morphology, orthography, and errors relevant
to the intended use. Report limitations of corpus coverage and reference-based
metrics alongside numerical results.

## Environment Acceptance

Authenticated Drive access, GPU execution, trained checkpoint compatibility,
container operation, and hosted access controls require tests in their target
environments. Deployment acceptance should include artifact integrity,
authentication, filesystem permissions, interface behavior, and recovery after
restart. These checks establish operational readiness, not model effectiveness.
