import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from twfr_mt.config import TranslationStep, checkpoint_path, load_routes
from twfr_mt.inference import LocalSeq2SeqTranslator, TranslationService


ROOT = Path(__file__).resolve().parents[1]


def service(loader):
    routes = load_routes(ROOT / "configs/models.example.json", ROOT / "models")
    return TranslationService(routes, ROOT / "models", loader=loader)


class RecordingTranslator:
    def __init__(self, path, calls):
        self.path, self.calls = path, calls

    def translate(self, text, step, max_new_tokens, num_beams):
        self.calls.append((text, step.source_lang, step.target_lang, max_new_tokens, num_beams))
        return f"{step.target_lang}:{text}"


def test_direct_and_pivot_keep_correct_language_order_and_reuse_models():
    calls, loads = [], []

    def loader(path):
        loads.append(path)
        return RecordingTranslator(path, calls)

    translator = service(loader)
    assert translator.translate("Bonjour", "fr-tw-pivot") == "tw:en:Bonjour"
    assert [(call[1], call[2]) for call in calls] == [("fr", "en"), ("en", "tw")]
    translator.translate("Encore", "fr-tw-pivot")
    assert len(loads) == 2
    translator.translate("Hello", "fr-tw-direct")
    assert len(translator.cache) == 2


@pytest.mark.parametrize("text", ["", "   ", "x" * 4001])
def test_input_bounds_reject_before_model_loading(text):
    translator = service(lambda path: pytest.fail("Should not load a model"))
    with pytest.raises(ValueError):
        translator.translate(text, "fr-tw-direct")


@pytest.mark.parametrize("kwargs", [{"num_beams": 0}, {"num_beams": 9}, {"max_new_tokens": 257}, {"max_new_tokens": True}])
def test_generation_bounds(kwargs):
    with pytest.raises(ValueError):
        service(lambda path: None).translate("Hello", "fr-tw-direct", **kwargs)


@pytest.mark.parametrize("path", ["../outside", "", ".", "C:/outside"])
def test_checkpoint_paths_are_bounded(path):
    with pytest.raises(ValueError):
        checkpoint_path(ROOT / "models", path)


def test_unicode_is_not_lowercased_or_stripped_of_accents():
    calls = []
    translator = service(lambda path: RecordingTranslator(path, calls))
    translator.translate("  E\u0301cole \u0254 \u025b  ", "fr-tw-direct")
    assert calls[0][0] == "\u00c9cole \u0254 \u025b"


def test_registry_rejects_duplicate_ids_and_broken_chains(tmp_path):
    original = json.loads((ROOT / "configs/models.example.json").read_text())
    original["routes"][1]["id"] = original["routes"][0]["id"]
    file = tmp_path / "registry.json"
    file.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="unique"):
        load_routes(file, ROOT / "models")
    original["routes"] = [original["routes"][2]]
    original["routes"][0]["steps"][1]["source_lang"] = "tw"
    file.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="chain"):
        load_routes(file, ROOT / "models")


def test_cached_tokenizer_restores_language_defaults_and_checks_saved_metadata():
    class Tokens:
        shape = (1, 2)

        def to(self, device):
            return self

    class Tokenizer:
        src_lang, tgt_lang = "original-source", "original-target"

        def get_vocab(self):
            return {"custom-source": 4, "custom-target": 5}

        def __call__(self, text, **kwargs):
            return {"input_ids": Tokens()}

        def decode(self, row, **kwargs):
            return "translated"

    translator = LocalSeq2SeqTranslator.__new__(LocalSeq2SeqTranslator)
    translator.tokenizer = Tokenizer()
    translator.model = SimpleNamespace(generate=lambda **kwargs: [[1, 2]])
    translator.torch = SimpleNamespace(inference_mode=nullcontext)
    translator.device = "cpu"
    translator.max_source_tokens = 8
    translator.language_defaults = {"src_lang": "original-source", "tgt_lang": "original-target"}
    translator.metadata = {}
    explicit = TranslationStep("tw", "fr", "example", "custom-source", "custom-target")
    translator.translate("hello", explicit, 8, 1)
    assert translator.tokenizer.src_lang == "custom-source"
    assert translator.tokenizer.tgt_lang == "custom-target"
    default = TranslationStep("tw", "fr", "example")
    translator.translate("hello", default, 8, 1)
    assert translator.tokenizer.src_lang == "original-source"
    assert translator.tokenizer.tgt_lang == "original-target"
    translator.metadata = {"source_lang": "en", "target_lang": "fr"}
    with pytest.raises(ValueError, match="saved language configuration"):
        translator.translate("hello", default, 8, 1)
