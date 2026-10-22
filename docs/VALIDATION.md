# Validation Record

Validated locally on Windows on 2026-10-22. This record describes software
verification, not translation quality or a completed corpus experiment.

## Completed Checks

- Full test suite: 146 passed, with one Transformers generation-configuration
  warning from the synthetic model fixture.
- Ruff checks: passed for `twfr_mt`, `tests`, and `scripts`.
- All three Colab notebooks: valid JSON; Python cells parsed after excluding
  notebook magics. They were not executed in Colab.
- Gradio: browser screenshots and horizontal-overflow checks passed at
  1280 x 900 and 390 x 844. No trained user checkpoint was loaded in the UI.
- Source bundle: ZIP integrity checked; no virtual environment, Git history,
  model weights, or Python bytecode included.

## Offline Model Integration

The real-dependency integration test creates a tiny random BART model and a
synthetic tokenizer locally. It trains on synthetic text, simulates interruption
after saving a full Trainer checkpoint, resumes optimization, exports
safetensors and tokenizer assets, and loads the final export for inference.
The test makes no network model download and uses no private Drive corpus.
Its forced output token checks pipeline execution, not learned translation.

The isolated CPU environment used Python 3.13, PyTorch 2.14.1+cpu,
Transformers 4.57.1, Datasets 4.8.5, Accelerate 1.15.0, and SacreBLEU 2.6.0.
The browser preview used Gradio 6.8.0 in a separate local Python environment.

## Outstanding Deployment Checks

Authenticated Drive inventory, actual corpus validation, full-scale training,
GPU execution in Colab, checkpoint language compatibility, translation quality,
Docker image execution, and hosted deployment remain unverified. Publishing
source code does not validate deployment, which requires explicit selection
and authorization of model artifacts and credentials.
