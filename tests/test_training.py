"""Dependency-free runner tests: no downloads, GPU, or actual optimization."""

from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from twfr_mt import training


@pytest.fixture
def config(tmp_path):
    paths = {}
    for name, text in (("train_source", "private train source\n"),
                       ("train_target", "private train target\n"),
                       ("validation_source", "private validation source\n"),
                       ("validation_target", "private validation target\n")):
        path = tmp_path / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        paths[name] = str(path)
    return training.TrainingConfig(base_model="example/model", output_dir=str(tmp_path / "run"),
                                   source_lang="tw", target_lang="fr", **paths)


def checkpoint(config, step=2):
    path = Path(config.output_dir) / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).touch()
    metadata = {"config": asdict(config), "files": {name: training._file_info(getattr(config, name)) for name in training._DATA_PATHS}}
    (path.parent / "run_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    return str(path)


@pytest.fixture
def runtime(monkeypatch):
    events = []
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.unk_token_id = 99
    tokenizer.src_lang = None
    tokenizer.tgt_lang = None
    tokenizer.get_vocab.return_value = {"twi_Latn": 11, "fra_Latn": 12, "<unk>": 99}
    token_calls = []

    def tokenize(text=None, **kwargs):
        token_calls.append((text, kwargs))
        rows = kwargs.get("text_target", text)
        return {"input_ids": [[4, 5] for _ in rows], "attention_mask": [[1, 1] for _ in rows]}

    tokenizer.side_effect = tokenize
    tokenizer.batch_decode.side_effect = lambda rows, **kwargs: ["decoded" for _ in rows]
    model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True, _commit_hash="abc123"),
                            generation_config=SimpleNamespace(forced_bos_token_id=None,
                                                              save_pretrained=Mock()))

    class Dataset:
        column_names = ["source", "target"]

        @classmethod
        def from_dict(cls, rows):
            obj = cls()
            obj.rows = rows
            return obj

        def map(self, callback, **kwargs):
            assert kwargs == {"batched": True, "remove_columns": self.column_names,
                              "load_from_cache_file": False}
            return callback(self.rows)

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state.trainer = self

        def train(self, **kwargs):
            assert (Path(state.config.output_dir) / "run_config.json").is_file()
            events.append(("train", kwargs))
            if state.fail:
                raise RuntimeError("simulated interruption")
            return SimpleNamespace(metrics={"train_loss": 0.5})

        def save_state(self):
            events.append(("save_state", {}))

        def evaluate(self, **kwargs):
            events.append(("evaluate", kwargs))
            return {"validation_bleu": 4.0}

        def save_model(self, directory):
            events.append(("save_model", directory))
            (Path(directory) / "model.safetensors").touch()

    hf = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                         AutoModelForSeq2SeqLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
                         set_seed=Mock(), Seq2SeqTrainingArguments=Mock(side_effect=lambda **kw: kw),
                         DataCollatorForSeq2Seq=Mock(side_effect=lambda **kw: kw), Seq2SeqTrainer=Trainer)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=False),
                                                  is_bf16_supported=Mock(return_value=False)))
    state = SimpleNamespace(torch=torch, transformers=hf, Dataset=Dataset,
                            versions={"transformers": "4.57.1", "torch": "2.6.0",
                                      "datasets": "3.0.0", "accelerate": "1.0.0", "sacrebleu": "2.3.1"},
                            tokenizer=tokenizer, model=model, events=events,
                            token_calls=token_calls, fail=False)
    monkeypatch.setattr(training, "_load_runtime", lambda: state)
    return state


def test_import_is_lazy():
    root = str(Path(__file__).resolve().parents[1])
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import twfr_mt.training; "
        "assert not {'torch', 'transformers', 'datasets', 'accelerate', 'sacrebleu'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code, root], check=True, capture_output=True, text=True)


def test_defaults_and_normalization(config):
    normalized = training.validate_config(config)
    assert normalized is not config
    assert normalized.epochs == 3
    assert normalized.max_steps == -1
    assert normalized.revision == "main"
    assert Path(normalized.output_dir).is_absolute()
    assert not Path(config.output_dir).exists()


@pytest.mark.parametrize("field,value", [
    ("epochs", float("nan")), ("epochs", float("inf")), ("epochs", 0), ("epochs", True),
    ("epochs", 10**400),
    ("learning_rate", float("-inf")), ("learning_rate", "0.01"), ("learning_rate", -1),
    ("batch_size", True), ("batch_size", 1.0), ("batch_size", 0),
    ("gradient_accumulation_steps", 0), ("max_source_length", 0),
    ("max_target_length", 1), ("max_target_length", float("inf")),
    ("save_total_limit", 0), ("max_steps", 0), ("max_steps", -2), ("max_steps", 1.0),
    ("seed", -1), ("seed", True), ("seed", 2**32), ("device", "mps"),
    ("source_lang", ""), ("target_lang", " "), ("base_model", ""), ("revision", ""),
    ("source_tokenizer_lang", ""), ("target_tokenizer_lang", 7),
    ("source_prefix", None), ("output_dir", ""), ("train_source", "missing-file"),
    ("resume_from_checkpoint", ""),
])
def test_invalid_config(config, field, value):
    with pytest.raises(ValueError):
        training.validate_config(replace(config, **{field: value}))


def test_nonempty_output_is_not_overwritten(config):
    output = Path(config.output_dir)
    output.mkdir()
    protected = output / "config.json"
    protected.write_text("pretrained model", encoding="utf-8")
    with pytest.raises(ValueError, match="pretrained export"):
        training.validate_config(config)
    assert protected.read_text(encoding="utf-8") == "pretrained model"


def test_resume_requires_full_checkpoint_inside_output(config, tmp_path):
    resume = checkpoint(config)
    valid = replace(config, resume_from_checkpoint=resume)
    assert training.validate_config(valid).resume_from_checkpoint == str(Path(resume).resolve())
    with pytest.raises(ValueError, match="output_dir/checkpoint"):
        training.validate_config(replace(valid, output_dir=str(tmp_path / "other")))
    (Path(resume) / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer.pt"):
        training.validate_config(valid)


def test_resume_cannot_reuse_pretrained_export(config):
    resume = checkpoint(config)
    (Path(config.output_dir) / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pretrained export"):
        training.validate_config(replace(config, resume_from_checkpoint=resume))


def test_nonempty_unrelated_output_is_rejected(config):
    output = Path(config.output_dir)
    output.mkdir()
    (output / "notes.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        training.validate_config(config)


@pytest.mark.parametrize("filename", ["trainer_state.json", "model.safetensors", "scheduler.pt", "rng_state.pth"])
def test_incomplete_resume_is_rejected(config, filename):
    resume = checkpoint(config)
    (Path(resume) / filename).unlink()
    with pytest.raises(ValueError):
        training.validate_config(replace(config, resume_from_checkpoint=resume))


def test_checkpoint_state_step_must_match(config):
    resume = checkpoint(config)
    (Path(resume) / "trainer_state.json").write_text('{"global_step": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid trainer state"):
        training.validate_config(replace(config, resume_from_checkpoint=resume))


def test_final_export_is_never_overwritten(config):
    resume = checkpoint(config)
    (Path(config.output_dir) / "final").mkdir()
    with pytest.raises(ValueError, match="final already exists"):
        training.validate_config(replace(config, resume_from_checkpoint=resume))


def test_runner_cpu_contract_and_private_metadata(config, runtime):
    runtime.config = replace(config, source_prefix="translate: ")
    result = training.run_training(runtime.config)
    args = runtime.trainer.kwargs["args"]
    assert args["use_cpu"] and not args["fp16"] and not args["bf16"]
    assert args["report_to"] == "none"
    assert args["eval_strategy"] == args["save_strategy"] == "epoch"
    assert args["predict_with_generate"] and args["save_safetensors"]
    assert args["save_only_model"] is False
    assert args["seed"] == args["data_seed"] == 42
    assert "processing_class" in runtime.trainer.kwargs
    assert "tokenizer" not in runtime.trainer.kwargs
    assert runtime.trainer.kwargs["data_collator"]["label_pad_token_id"] == -100
    assert runtime.trainer.kwargs["data_collator"]["padding"] is True
    assert runtime.token_calls[0][0] == ["translate: private train source"]
    assert runtime.token_calls[0][1]["max_length"] == 128
    assert runtime.token_calls[1][1]["text_target"] == ["private train target"]
    assert all(call[1]["padding"] is False for call in runtime.token_calls)
    assert runtime.events[0] == ("train", {"resume_from_checkpoint": None})
    assert ("evaluate", {"metric_key_prefix": "validation"}) in runtime.events
    assert runtime.tokenizer.save_pretrained.call_count == 1
    assert Path(runtime.tokenizer.save_pretrained.call_args.args[0]).name.startswith(".final-export-")
    assert (Path(result["final_dir"]) / "export_complete.json").is_file()
    assert (Path(result["final_dir"]) / "model.safetensors").is_file()
    raw = Path(result["run_config"]).read_text(encoding="utf-8")
    assert "private train source" not in raw
    assert "private validation target" not in raw
    document = json.loads(raw)
    assert document["splits"]["train"]["pairs"] == 1
    assert document["versions"]["transformers"] == "4.57.1"
    assert document["model"]["resolved_revision"] == "abc123"
    assert document["files"]["train_source"]["sha256"] == hashlib.sha256(
        Path(config.train_source).read_bytes()).hexdigest()
    assert result["validation_metrics"] == {"validation_bleu": 4.0}
    assert not (Path(config.output_dir) / ".training.lock").exists()


@pytest.mark.parametrize("bf16", [True, False])
def test_cuda_precision(config, runtime, bf16):
    runtime.config = replace(config, device="cuda")
    runtime.torch.cuda.is_available.return_value = True
    runtime.torch.cuda.is_bf16_supported.return_value = bf16
    training.run_training(runtime.config)
    args = runtime.trainer.kwargs["args"]
    assert args["bf16"] is bf16
    assert args["fp16"] is not bf16
    assert not args["use_cpu"]


def test_explicit_cpu_ignores_cuda(config, runtime):
    runtime.config = replace(config, device="cpu")
    runtime.torch.cuda.is_available.return_value = True
    training.run_training(runtime.config)
    assert runtime.trainer.kwargs["args"]["use_cpu"]
    runtime.torch.cuda.is_bf16_supported.assert_not_called()


def test_cuda_unavailable(config, runtime):
    with pytest.raises(ValueError, match="CUDA"):
        training.run_training(replace(config, device="cuda"))
    assert not Path(config.output_dir).exists()


def test_resume_forwards_state_path(config, runtime):
    runtime.config = replace(config, resume_from_checkpoint=checkpoint(config))
    training.run_training(runtime.config)
    assert runtime.events[0] == ("train", {"resume_from_checkpoint": runtime.config.resume_from_checkpoint})


def test_existing_metadata_must_match(config, runtime):
    resume = checkpoint(config)
    metadata = Path(config.output_dir) / "run_config.json"
    metadata.write_text('{"config": {}, "files": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        training.run_training(replace(config, resume_from_checkpoint=resume))
    assert metadata.read_text(encoding="utf-8") == '{"config": {}, "files": {}}'


def test_interruption_preserves_metadata_and_releases_lock(config, runtime):
    runtime.config = config
    runtime.fail = True
    with pytest.raises(RuntimeError, match="interruption"):
        training.run_training(config)
    assert (Path(config.output_dir) / "run_config.json").is_file()
    assert not (Path(config.output_dir) / ".training.lock").exists()
    assert not (Path(config.output_dir) / "final").exists()


def test_split_leakage_fails_before_model_import(config, runtime):
    Path(config.validation_source).write_text("PRIVATE TRAIN SOURCE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        training.run_training(config)
    runtime.transformers.AutoTokenizer.from_pretrained.assert_not_called()


def test_language_tokens_only_when_explicit(config, runtime):
    runtime.config = config
    training.run_training(config)
    assert runtime.model.generation_config.forced_bos_token_id is None
    runtime.tokenizer.get_vocab.assert_not_called()


def test_explicit_multilingual_tokens(config, runtime):
    runtime.config = replace(config, source_tokenizer_lang="twi_Latn", target_tokenizer_lang="fra_Latn")
    training.run_training(runtime.config)
    assert runtime.tokenizer.src_lang == "twi_Latn"
    assert runtime.tokenizer.tgt_lang == "fra_Latn"
    assert runtime.model.generation_config.forced_bos_token_id == 12


@pytest.mark.parametrize("token", ["fr", "<unk>"])
def test_unknown_language_token_fails(config, runtime, token):
    with pytest.raises(ValueError, match="vocabulary token"):
        training.run_training(replace(config, target_tokenizer_lang=token))


def test_marian_does_not_invent_language_properties(config, runtime):
    del runtime.tokenizer.tgt_lang
    with pytest.raises(ValueError, match="does not support"):
        training.run_training(replace(config, target_tokenizer_lang="fra_Latn"))


def test_metric_callback_unmasks_and_keeps_only_numeric_metrics(runtime, monkeypatch):
    metrics = importlib.import_module("twfr_mt.metrics")
    scorer = Mock(return_value={"bleu": 1, "chrf": 2.5, "ter": 3, "signatures": {"bleu": "signature"}})
    monkeypatch.setattr(metrics, "score_translations", scorer)
    runtime.tokenizer.batch_decode.side_effect = [[" prediction ", ""], [" reference ", " other "]]
    prediction = SimpleNamespace(predictions=([[3, -100], [0, 0]], "ignored"),
                                 label_ids=[[4, -100], [5, 0]])
    result = training._metric_callback(runtime.tokenizer)(prediction)
    assert result == {"bleu": 1.0, "chrf": 2.5, "ter": 3.0}
    assert runtime.tokenizer.batch_decode.call_args_list[0].args[0] == [[3, 0], [0, 0]]
    assert runtime.tokenizer.batch_decode.call_args_list[1].args[0] == [[4, 0], [5, 0]]
    scorer.assert_called_once_with(["prediction", ""], ["reference", "other"])


@pytest.mark.parametrize("versions", [
    {"transformers": "5.0.0"}, {"transformers": "4.57.0"}, {"torch": "2.5.1"},
    {"torch": "3.0.0"}, {"datasets": "2.9.0"}, {"datasets": "5.0.0"},
    {"accelerate": "0.34.0"}, {"accelerate": "2.0.0"},
])
def test_runtime_rejects_unsupported_versions(monkeypatch, versions):
    pytest.importorskip("packaging.version")
    installed = {"transformers": "4.57.1", "torch": "2.6.0", "datasets": "3.0.0",
                 "accelerate": "1.0.0", "sacrebleu": "2.3.1", **versions}
    monkeypatch.setattr(training.metadata, "version", installed.__getitem__)
    with pytest.raises(RuntimeError, match="Use"):
        training._load_runtime()


def test_atomic_json_failure_leaves_original_and_no_temp_files(tmp_path):
    path = tmp_path / "run_config.json"
    path.write_text('{"safe": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        training._atomic_json(path, {"not_finite": float("nan")})
    assert path.read_text(encoding="utf-8") == '{"safe": true}'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("entrypoint", [training.validate_config, training.run_training])
def test_resume_requires_original_run_config(config, runtime, entrypoint):
    resume = checkpoint(config)
    output = Path(config.output_dir)
    (output / "run_config.json").unlink()
    state_before = (Path(resume) / "trainer_state.json").read_bytes()

    with pytest.raises(ValueError, match="original run_config.json provenance"):
        entrypoint(replace(config, resume_from_checkpoint=resume))

    runtime.transformers.AutoTokenizer.from_pretrained.assert_not_called()
    runtime.transformers.AutoModelForSeq2SeqLM.from_pretrained.assert_not_called()
    assert runtime.events == []
    assert not (output / "run_config.json").exists()
    assert not (output / "final").exists()
    assert (Path(resume) / "trainer_state.json").read_bytes() == state_before


def test_resume_rejects_changed_resolved_model_revision(config, runtime):
    resume = checkpoint(config)
    runtime.config = replace(config, resume_from_checkpoint=resume)
    output = Path(config.output_dir)
    metadata_path = output / "run_config.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["model"] = {"requested_revision": config.revision, "resolved_revision": "older-commit"}
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    metadata_before = metadata_path.read_bytes()
    assert runtime.model.config._commit_hash != document["model"]["resolved_revision"]

    with pytest.raises(ValueError, match="does not match"):
        training.run_training(runtime.config)

    runtime.transformers.AutoModelForSeq2SeqLM.from_pretrained.assert_called_once()
    runtime.transformers.Seq2SeqTrainingArguments.assert_not_called()
    assert runtime.events == []
    assert metadata_path.read_bytes() == metadata_before
    assert not (output / ".training.lock").exists()
    assert not (output / "final").exists()
    assert not list(output.glob(".final-export-*"))


def test_tokenizer_export_failure_cleans_staging_without_publishing_final(config, runtime):
    resume = checkpoint(config)
    runtime.config = replace(config, resume_from_checkpoint=resume)
    output = Path(config.output_dir)
    checkpoint_before = {path.name: path.read_bytes() for path in Path(resume).iterdir()}

    def fail_tokenizer_export(directory):
        staging = Path(directory)
        assert staging.parent == output
        assert staging.name.startswith(".final-export-")
        assert (staging / "model.safetensors").is_file()
        assert not (output / "final").exists()
        (staging / "tokenizer_config.json").write_text("partial export", encoding="utf-8")
        raise OSError("simulated tokenizer export failure")

    runtime.tokenizer.save_pretrained.side_effect = fail_tokenizer_export
    with pytest.raises(OSError, match="simulated tokenizer export failure"):
        training.run_training(runtime.config)

    runtime.tokenizer.save_pretrained.assert_called_once()
    runtime.model.generation_config.save_pretrained.assert_not_called()
    assert not (output / "final").exists()
    assert not list(output.glob(".final-export-*"))
    assert not (output / ".training.lock").exists()
    assert not (output / "training_summary.json").exists()
    assert json.loads((output / "run_config.json").read_text(encoding="utf-8"))["model"][
        "resolved_revision"] == runtime.model.config._commit_hash
    assert {path.name: path.read_bytes() for path in Path(resume).iterdir()} == checkpoint_before


def test_exported_generation_config_clears_inherited_max_new_tokens(config, runtime):
    runtime.config = replace(config, max_target_length=32)
    generation = runtime.model.generation_config
    generation.max_new_tokens = 999
    generation.max_length = 20

    def save_generation_config(directory):
        path = Path(directory) / "generation_config.json"
        path.write_text(json.dumps({"max_length": generation.max_length,
                                    "max_new_tokens": generation.max_new_tokens}), encoding="utf-8")

    generation.save_pretrained.side_effect = save_generation_config
    result = training.run_training(runtime.config)

    arguments = runtime.trainer.kwargs["args"]
    assert arguments["generation_max_length"] == 32
    assert arguments["generation_config"].max_new_tokens is None
    generation.save_pretrained.assert_called_once()
    saved = json.loads((Path(result["final_dir"]) / "generation_config.json").read_text(encoding="utf-8"))
    assert saved == {"max_length": 32, "max_new_tokens": None}
    assert (Path(result["final_dir"]) / "export_complete.json").is_file()
