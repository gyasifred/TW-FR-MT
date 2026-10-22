"""Corpus translation metrics with explicit, reproducible SacreBLEU settings."""

from __future__ import annotations

from collections.abc import Iterable


def _sentences(values: Iterable[str], name: str, *, allow_blank: bool = False) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of strings, not a single string")
    rows = list(values)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    for row, value in enumerate(rows, 1):
        if not isinstance(value, str):
            raise TypeError(f"{name} row {row} must be a string")
        if not allow_blank and not value.strip():
            raise ValueError(f"{name} row {row} must not be blank")
    return rows


def score_translations(predictions: Iterable[str], references: Iterable[str]) -> dict:
    """Score one aligned reference per prediction, without altering either.

    Returns numeric ``bleu``, ``chrf``, ``ter`` scores and a ``signatures`` map
    with the same three keys. BLEU uses SacreBLEU's 13a tokenizer and exponential
    smoothing; chrF uses character order 6, word order 0, beta 2. TER is explicitly
    case-sensitive with normalization/punctuation removal disabled. Standard
    metric tokenization still applies, as recorded in the signatures. All
    hypotheses are passed through unchanged, including empty/whitespace-only
    hypotheses: they count as translation errors, not missing rows. Empty
    corpora, blank references, or unequal lengths fail before importing SacreBLEU.
    """
    predictions = _sentences(predictions, "predictions", allow_blank=True)
    references = _sentences(references, "references")
    if len(predictions) != len(references):
        raise ValueError(f"Unequal lengths: {len(predictions)} predictions, {len(references)} references")
    try:
        from sacrebleu.metrics import BLEU, CHRF, TER
    except ImportError as exc:
        raise ImportError("score_translations requires sacrebleu; install it with: pip install sacrebleu") from exc
    metrics = {
        "bleu": BLEU(lowercase=False, tokenize="13a", smooth_method="exp", effective_order=False),
        "chrf": CHRF(char_order=6, word_order=0, beta=2, lowercase=False),
        "ter": TER(normalized=False, no_punct=False, asian_support=False, case_sensitive=True),
    }
    scores, signatures = {}, {}
    for name, metric in metrics.items():
        scores[name] = float(metric.corpus_score(predictions, [references]).score)
        signatures[name] = str(metric.get_signature())
    return {**scores, "signatures": signatures}
