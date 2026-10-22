"""Metric contracts run without SacreBLEU; integration tests skip if absent."""

import builtins
import sys
from types import ModuleType

import pytest

from twfr_mt.metrics import score_translations


@pytest.mark.parametrize("predictions,references,error", [
    ([], [], ValueError), (["a"], [], ValueError), ([], ["a"], ValueError),
    (["a", "b"], ["a"], ValueError), (["a"], [""], ValueError),
    (["a"], ["\t\u00a0"], ValueError), ([None], ["a"], TypeError),
    (["a"], [1], TypeError), ("text", ["text"], TypeError),
    (["text"], "text", TypeError), (b"text", ["text"], TypeError),
])
def test_invalid_input_fails_before_dependency_import(monkeypatch, predictions, references, error):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("sacrebleu"):
            pytest.fail("invalid input must fail before importing sacrebleu")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(error):
        score_translations(predictions, references)


def test_missing_dependency_has_actionable_error(monkeypatch):
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name.startswith("sacrebleu"):
            raise ModuleNotFoundError("No module named sacrebleu")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="pip install sacrebleu"):
        score_translations(["a"], ["b"])


@pytest.mark.parametrize("predictions", [
    ["  E\u0301COLE\t!  "], [""], [" \t "], ["", " \t ", "  E\u0301COLE\t!  "],
])
def test_inputs_unmodified_and_configuration_explicit(monkeypatch, predictions):
    calls = {}
    fake = ModuleType("sacrebleu.metrics")
    def factory(name):
        class Metric:
            def __init__(self, **kwargs):
                calls[name] = {"settings": kwargs}
            def corpus_score(self, predictions, references):
                calls[name]["predictions"] = predictions[:]
                calls[name]["references"] = [references[0][:]]
                return type("Score", (), {"score": 12.5})()
            def get_signature(self):
                assert "predictions" in calls[name]
                return name + " signature"
        return Metric
    for name in ("BLEU", "CHRF", "TER"):
        setattr(fake, name, factory(name))
    monkeypatch.setitem(sys.modules, "sacrebleu.metrics", fake)
    references = ["\u00e9cole !"] * len(predictions)
    result = score_translations(iter(predictions), iter(references))
    assert result == {"bleu": 12.5, "chrf": 12.5, "ter": 12.5, "signatures": {"bleu": "BLEU signature", "chrf": "CHRF signature", "ter": "TER signature"}}
    for call in calls.values():
        assert call["predictions"] == predictions
        assert call["references"] == [references]
    assert calls["BLEU"]["settings"]["lowercase"] is False
    assert calls["CHRF"]["settings"]["lowercase"] is False
    assert calls["TER"]["settings"] == {"normalized": False, "no_punct": False, "asian_support": False, "case_sensitive": True}


def test_sacrebleu_identical_corpus_and_signatures():
    pytest.importorskip("sacrebleu")
    text = ["This is a sufficiently long sentence with accents caf\u00e9."]
    result = score_translations(text, text)
    assert result["bleu"] == pytest.approx(100)
    assert result["chrf"] == pytest.approx(100)
    assert result["ter"] == pytest.approx(0)
    assert set(result["signatures"]) == {"bleu", "chrf", "ter"}
    assert all("version:" in signature for signature in result["signatures"].values())
    assert "case:mixed" in result["signatures"]["bleu"]
    assert "case:mixed" in result["signatures"]["chrf"]
    assert "case:mixed" in result["signatures"]["ter"]


def test_sacrebleu_matches_direct_api_and_preserves_case():
    pytest.importorskip("sacrebleu")
    from sacrebleu.metrics import BLEU, CHRF, TER
    predictions = ["This IS a rather long sentence with E\u0301cole !"]
    references = ["this is a rather long sentence with \u00e9cole!"]
    result = score_translations(predictions, references)
    for name, metric in {
        "bleu": BLEU(lowercase=False, tokenize="13a", smooth_method="exp", effective_order=False),
        "chrf": CHRF(char_order=6, word_order=0, beta=2, lowercase=False),
        "ter": TER(normalized=False, no_punct=False, asian_support=False, case_sensitive=True),
    }.items():
        assert result[name] == pytest.approx(metric.corpus_score(predictions, [references]).score)
        assert result["signatures"][name] == str(metric.get_signature())
    assert result["bleu"] < 100
    assert result["chrf"] < 100
    assert result["ter"] > 0


@pytest.mark.parametrize("predictions", [[""], [" \t "], ["", " \t "]])
def test_sacrebleu_all_empty_hypotheses_are_real_errors(predictions):
    pytest.importorskip("sacrebleu")
    references = ["This reference contains several real words."] * len(predictions)
    result = score_translations(predictions, references)
    assert result["bleu"] == pytest.approx(0)
    assert result["chrf"] == pytest.approx(0)
    assert result["ter"] == pytest.approx(100)


def test_sacrebleu_mixed_empty_hypotheses_not_dropped():
    pytest.importorskip("sacrebleu")
    text = "This reference contains several real words."
    result = score_translations([text, ""], [text, text])
    assert 0 < result["bleu"] < 100
    assert 0 < result["chrf"] < 100
    assert result["ter"] == pytest.approx(50)
