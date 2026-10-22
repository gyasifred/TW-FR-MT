"""Colab seq2seq training, targeting transformers==4.57.1 (never v5).

Keep the source checkout on Colab's local disk and put ``output_dir`` on a
mounted Drive. Only train and validation splits enter this runner. Importing
this module needs no ML packages; install the supported stack in Colab first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
from uuid import uuid4


@dataclass
class TrainingConfig:
    base_model: str
    train_source: str
    train_target: str
    validation_source: str
    validation_target: str
    output_dir: str
    source_lang: str
    target_lang: str
    revision: str = "main"
    epochs: float = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    max_source_length: int = 128
    max_target_length: int = 128
    seed: int = 42
    resume_from_checkpoint: str | None = None
    source_tokenizer_lang: str | None = None
    target_tokenizer_lang: str | None = None
    source_prefix: str = ""
    save_total_limit: int = 2
    max_steps: int = -1
    device: str = "auto"


_DATA_PATHS = ("train_source", "train_target", "validation_source", "validation_target")
_RESUME_MUTABLE = {"resume_from_checkpoint", "device", "epochs", "max_steps", "save_total_limit"}


def _string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string without NUL characters")


def _checkpoint(path: Path, output: Path) -> None:
    if path.parent != output or not re.fullmatch(r"checkpoint-[0-9]+", path.name):
        raise ValueError("resume_from_checkpoint must be an output_dir/checkpoint-<step> directory")
    if not path.is_dir() or not (path / "trainer_state.json").is_file():
        raise ValueError("resume checkpoint must contain trainer_state.json")
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        step = state["global_step"]
        if type(step) is not int or step != int(path.name.split("-")[1]) or step < 1:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("resume checkpoint has invalid trainer state") from exc
    weights = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
               "pytorch_model.bin.index.json")
    if not any((path / name).is_file() for name in weights):
        raise ValueError("resume checkpoint has no model weights")
    for name in ("optimizer.pt", "scheduler.pt"):
        if not (path / name).is_file():
            raise ValueError(f"resume checkpoint is missing {name}; use a full Trainer checkpoint")
    if not any(path.glob("rng_state*.pth")):
        raise ValueError("resume checkpoint is missing RNG state")


def validate_config(config: TrainingConfig) -> TrainingConfig:
    """Return a normalized copy, without creating files or importing ML packages.

    Existing output is allowed only with a complete checkpoint directly inside
    that output directory. A finished ``final`` export is never overwritten.
    """
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    for name in ("base_model", "revision", "source_lang", "target_lang", "device"):
        _string(name, getattr(config, name))
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if not isinstance(config.source_prefix, str) or "\x00" in config.source_prefix:
        raise ValueError("source_prefix must be a string without NUL characters")
    for name in ("source_tokenizer_lang", "target_tokenizer_lang"):
        if getattr(config, name) is not None:
            _string(name, getattr(config, name))
    for name in ("epochs", "learning_rate"):
        value = getattr(config, name)
        try:
            valid = (not isinstance(value, bool) and isinstance(value, Real)
                     and math.isfinite(value) and value > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("batch_size", "gradient_accumulation_steps", "max_source_length",
                 "max_target_length", "save_total_limit", "seed", "max_steps"):
        value = getattr(config, name)
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        minimum = 0 if name == "seed" else 1
        if name == "max_steps":
            if value != -1 and value < 1:
                raise ValueError("max_steps must be -1 or a positive integer")
        elif value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if config.seed >= 2**32:
        raise ValueError("seed must be less than 2**32")
    if config.max_target_length < 2:
        raise ValueError("max_target_length must be >= 2 to generate beyond the decoder start token")
    paths = {}
    base_path = Path(config.base_model).expanduser()
    if base_path.is_dir():
        paths["base_model"] = str(base_path.resolve())
    elif base_path.is_absolute() or config.base_model.startswith("~"):
        raise ValueError("base_model local directory does not exist")
    for name in (*_DATA_PATHS, "output_dir"):
        _string(name, getattr(config, name))
        paths[name] = str(Path(getattr(config, name)).expanduser().resolve())
    for name in _DATA_PATHS:
        if not Path(paths[name]).is_file():
            raise ValueError(f"{name} must be an existing file")
    output = Path(paths["output_dir"])
    if output.exists() and not output.is_dir():
        raise ValueError("output_dir must be a directory")
    if (output / "config.json").exists() or any(output.glob("*.safetensors")):
        raise ValueError("output_dir looks like a pretrained export; choose a separate run directory")
    if (output / "final").exists() or (output / "final").is_symlink():
        raise ValueError("output_dir/final already exists; choose a new output directory")
    resume = config.resume_from_checkpoint
    if resume is not None:
        _string("resume_from_checkpoint", resume)
        checkpoint = Path(resume).expanduser().resolve()
        _checkpoint(checkpoint, output)
        if not (output / "run_config.json").is_file():
            raise ValueError("Resume requires the original run_config.json provenance")
        paths["resume_from_checkpoint"] = str(checkpoint)
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output_dir is nonempty; provide an explicit full checkpoint to resume")
    return replace(config, **paths)


def _load_runtime() -> SimpleNamespace:
    """Reject API drift before importing the training stack."""
    try:
        from packaging.version import Version

        versions = {name: metadata.version(name) for name in
                    ("transformers", "torch", "datasets", "accelerate", "sacrebleu")}
        if Version(versions["transformers"]) != Version("4.57.1"):
            raise RuntimeError("Use transformers==4.57.1, not transformers v5")
        for name, lower, upper in (("torch", "2.6", "3"), ("datasets", "3", "5"),
                                   ("accelerate", "1", "2")):
            if not Version(lower) <= Version(versions[name]) < Version(upper):
                raise RuntimeError(f"Use {name}>={lower},<{upper}")
        import torch
        import transformers
        from datasets import Dataset
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError("Install transformers==4.57.1 torch>=2.6,<3 datasets>=3,<5 "
                           "accelerate>=1,<2 plus sacrebleu and the model's tokenizer dependencies") from exc
    return SimpleNamespace(torch=torch, transformers=transformers, Dataset=Dataset, versions=versions)


def _file_info(path: str) -> dict:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path, "bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def _split_info(pairs: list) -> dict:
    return {"pairs": len(pairs),
            "source_characters": sum(len(pair.source) for pair in pairs),
            "target_characters": sum(len(pair.target) for pair in pairs)}


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _check_provenance(path: Path, document: dict) -> None:
    if not path.exists():
        return
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        old_config = previous["config"]
        if not isinstance(old_config, dict):
            raise ValueError
        expected = {k: v for k, v in document["config"].items() if k not in _RESUME_MUTABLE}
        actual = {k: v for k, v in old_config.items() if k not in _RESUME_MUTABLE}
        if expected != actual or previous["files"] != document["files"]:
            raise ValueError
        old_revision = previous.get("model", {}).get("resolved_revision")
        new_revision = document.get("model", {}).get("resolved_revision")
        if old_revision and new_revision and old_revision != new_revision:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Existing run_config.json does not match this run's data/configuration") from exc


def _configure_languages(tokenizer: Any, model: Any, config: TrainingConfig) -> None:
    vocab = None
    for field, attribute in (("source_tokenizer_lang", "src_lang"),
                             ("target_tokenizer_lang", "tgt_lang")):
        language = getattr(config, field)
        if language is None:
            continue
        if not hasattr(tokenizer, attribute):
            raise ValueError(f"This tokenizer does not support explicit {attribute}")
        if vocab is None:
            vocab = tokenizer.get_vocab()
        if language not in vocab or vocab[language] == tokenizer.unk_token_id:
            raise ValueError(f"{field} must name an existing non-UNK vocabulary token")
        setattr(tokenizer, attribute, language)
    if config.target_tokenizer_lang is not None:
        model.generation_config.forced_bos_token_id = vocab[config.target_tokenizer_lang]


def _metric_callback(tokenizer: Any):
    def compute_metrics(prediction: Any) -> dict:
        from .metrics import score_translations

        predictions = prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pad = tokenizer.pad_token_id
        # Trainer may pad generated sequences as well as labels with -100.
        decoded = tokenizer.batch_decode(
            [[pad if int(token) == -100 else int(token) for token in row] for row in predictions],
            skip_special_tokens=True,
        )
        references = tokenizer.batch_decode(
            [[pad if int(token) == -100 else int(token) for token in row]
             for row in prediction.label_ids], skip_special_tokens=True,
        )
        # Empty hypotheses are real translation errors, never synthetic tokens.
        scores = score_translations([text.strip() for text in decoded],
                                    [text.strip() for text in references])
        return {key: float(value) for key, value in scores.items()
                if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)}
    return compute_metrics


def run_training(config: TrainingConfig) -> dict:
    """Train, evaluate held-out validation, export safetensors, and return paths/metrics.

    No test data is accepted. Resume paths restore full Trainer state, not just
    pretrained weights. The base model is always loaded with the given revision.
    No credentials, environment variables, or corpus text enter run_config.json.
    """
    config = validate_config(config)
    from .data import read_parallel, validate_splits

    files = {name: _file_info(getattr(config, name)) for name in _DATA_PATHS}
    train = read_parallel(config.train_source, config.train_target)
    validation = read_parallel(config.validation_source, config.validation_target)
    validate_splits({"train": train, "validation": validation})
    if not train or not validation:
        raise ValueError("train and validation splits must both be nonempty")
    if files != {name: _file_info(getattr(config, name)) for name in _DATA_PATHS}:
        raise ValueError("Input data changed while being read")
    _check_provenance(Path(config.output_dir) / "run_config.json",
                      {"config": asdict(config), "files": files})
    runtime = _load_runtime()
    torch, hf = runtime.torch, runtime.transformers
    cuda = config.device != "cpu" and torch.cuda.is_available()
    if config.device == "cuda" and not cuda:
        raise ValueError("CUDA was requested but is unavailable")
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(cuda and not bf16)
    hf.set_seed(config.seed)
    tokenizer = hf.AutoTokenizer.from_pretrained(config.base_model, revision=config.revision,
                                                 trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError("The seq2seq tokenizer must define a pad token")
    model = hf.AutoModelForSeq2SeqLM.from_pretrained(config.base_model, revision=config.revision,
                                                   trust_remote_code=False)
    if not model.config.is_encoder_decoder:
        raise ValueError("base_model must be an encoder-decoder seq2seq model")
    _configure_languages(tokenizer, model, config)
    model.generation_config.max_length = config.max_target_length
    model.generation_config.max_new_tokens = None

    def tokenize(batch: dict) -> dict:
        inputs = tokenizer([config.source_prefix + text for text in batch["source"]],
                           truncation=True, max_length=config.max_source_length, padding=False)
        labels = tokenizer(text_target=batch["target"], truncation=True,
                           max_length=config.max_target_length, padding=False)
        inputs["labels"] = labels["input_ids"]
        return inputs

    def dataset(pairs: list):
        raw = runtime.Dataset.from_dict({"source": [pair.source for pair in pairs],
                                         "target": [pair.target for pair in pairs]})
        return raw.map(tokenize, batched=True, remove_columns=raw.column_names,
                       load_from_cache_file=False)

    train_dataset, eval_dataset = dataset(train), dataset(validation)
    output = Path(config.output_dir)
    document = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "files": files,
        "splits": {"train": _split_info(train), "validation": _split_info(validation)},
        "versions": {**runtime.versions, "python": sys.version.split()[0]},
        "model": {"requested_revision": config.revision,
                  "resolved_revision": getattr(model.config, "_commit_hash", None)},
        "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
        "device": "cuda" if cuda else "cpu",
        "metrics": {"split": "validation", "empty_prediction": "preserved_empty"},
    }
    output.mkdir(parents=True, exist_ok=True)
    _check_provenance(output / "run_config.json", document)
    # A second check catches files created while a model was downloading.
    validate_config(config)
    lock = output / ".training.lock"
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("Output is locked by another run; inspect .training.lock before retrying") from exc
    os.close(lock_descriptor)
    try:
        if config.resume_from_checkpoint is None and any(entry != lock for entry in output.iterdir()):
            raise ValueError("output_dir became nonempty before acquiring the run lock")
        if (output / "final").exists() or (output / "final").is_symlink():
            raise ValueError("output_dir/final already exists; refusing to overwrite it")
        _check_provenance(output / "run_config.json", document)
        _atomic_json(output / "run_config.json", document)
        arguments = hf.Seq2SeqTrainingArguments(
            output_dir=str(output), overwrite_output_dir=False,
            num_train_epochs=config.epochs, per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, seed=config.seed, data_seed=config.seed,
            max_steps=config.max_steps, save_total_limit=config.save_total_limit,
            eval_strategy="epoch", save_strategy="epoch", predict_with_generate=True,
            generation_max_length=config.max_target_length, generation_config=model.generation_config,
            save_safetensors=True, save_only_model=False,
            report_to="none", use_cpu=not cuda, fp16=fp16, bf16=bf16,
            dataloader_pin_memory=cuda,
        )
        trainer = hf.Seq2SeqTrainer(
            model=model, args=arguments, train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=hf.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                                   padding=True, label_pad_token_id=-100),
            compute_metrics=_metric_callback(tokenizer),
        )
        result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        trainer.save_state()
        evaluation = trainer.evaluate(metric_key_prefix="validation")
        final = output / "final"
        staging = output / f".final-export-{uuid4().hex}"
        staging.mkdir(exist_ok=False)
        try:
            trainer.save_model(str(staging))
            tokenizer.save_pretrained(str(staging))
            model.generation_config.save_pretrained(str(staging))
            if not any(staging.glob("*.safetensors")):
                raise RuntimeError("Final export did not produce safetensors weights")
            _atomic_json(staging / "twfr_metadata.json", {
                "schema_version": 1, "source_lang": config.source_lang,
                "target_lang": config.target_lang, "source_prefix": config.source_prefix,
                "source_tokenizer_lang": config.source_tokenizer_lang,
                "target_tokenizer_lang": config.target_tokenizer_lang,
                "max_source_length": config.max_source_length,
                "max_target_length": config.max_target_length,
            })
            _atomic_json(staging / "export_complete.json", {
                "schema_version": 1, "files": sorted(path.name for path in staging.iterdir()),
            })
            if final.exists():
                raise ValueError("Final export appeared during training; refusing to overwrite it")
            staging.rename(final)
        finally:
            if staging.exists() and staging.parent == output and staging.name.startswith(".final-export-"):
                shutil.rmtree(staging)
        summary = {"output_dir": str(output), "final_dir": str(final),
                   "run_config": str(output / "run_config.json"),
                   "train_metrics": result.metrics, "validation_metrics": evaluation}
        _atomic_json(output / "training_summary.json", summary)
        return summary
    finally:
        lock.unlink(missing_ok=True)
