"""Offline tiny-model integration: plumbing evidence, not translation quality."""

from dataclasses import replace
import json

import pytest


def test_real_tiny_training_interruption_resume_export_and_local_inference(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    pytest.importorskip("sacrebleu")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from twfr_mt.config import load_routes
    from twfr_mt.inference import TranslationService
    from twfr_mt.training import TrainingConfig, run_training

    torch.set_num_threads(1)
    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "hello": 4,
                  "bonjour": 5, "one": 6, "two": 7, "three": 8, "four": 9}
    raw_tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = hf.PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>", model_input_names=["input_ids", "attention_mask"])
    model = hf.BartForConditionalGeneration(hf.BartConfig(
        vocab_size=len(vocabulary), d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
        forced_bos_token_id=4, forced_eos_token_id=2,
    ))
    base = tmp_path / "base"
    model.save_pretrained(base, safe_serialization=True)
    tokenizer.save_pretrained(base)
    files = {}
    for name, text in {
        "train_source": "hello one\nhello two\n",
        "train_target": "bonjour one\nbonjour two\n",
        "validation_source": "hello three\nhello four\n",
        "validation_target": "bonjour three\nbonjour four\n",
    }.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        files[name] = str(path)
    config = TrainingConfig(base_model=str(base), output_dir=str(tmp_path / "run"), source_lang="en", target_lang="fr", batch_size=1, gradient_accumulation_steps=1, epochs=1, max_steps=2, max_source_length=16, max_target_length=8, device="cpu", **files)
    save_state = hf.Seq2SeqTrainer.save_state

    def interrupted_save_state(trainer):
        save_state(trainer)
        raise RuntimeError("Simulated interruption after full Trainer checkpoint")

    monkeypatch.setattr(hf.Seq2SeqTrainer, "save_state", interrupted_save_state)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        run_training(config)
    checkpoint = tmp_path / "run" / "checkpoint-2"
    assert (checkpoint / "optimizer.pt").is_file()
    assert not (tmp_path / "run" / ".training.lock").exists()
    monkeypatch.setattr(hf.Seq2SeqTrainer, "save_state", save_state)
    summary = run_training(replace(config, max_steps=4, resume_from_checkpoint=str(checkpoint)))
    final = tmp_path / "run" / "final"
    assert (final / "model.safetensors").is_file()
    assert summary["validation_metrics"]["validation_loss"] >= 0
    assert json.loads((tmp_path / "run" / "trainer_state.json").read_text())["global_step"] == 4
    registry = tmp_path / "routes.json"
    registry.write_text(json.dumps({"routes": [{"id": "test", "label": "Offline smoke test", "source_lang": "en", "target_lang": "fr", "steps": [{"source_lang": "en", "target_lang": "fr", "checkpoint": "run/final"}]}]}))
    translator = TranslationService(load_routes(registry, tmp_path), tmp_path, device="cpu")
    assert translator.readiness("test")
    translated = translator.translate("hello", "test", max_new_tokens=8, num_beams=1)
    assert isinstance(translated, str) and "hello" in translated
