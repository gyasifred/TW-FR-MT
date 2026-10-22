"""Strict parallel-text IO, leakage audits, and reproducible grouped splits.

Text is NFC with Unicode whitespace collapsed, never lowercased. Only source
grouping keys are casefolded. Row numbers in audit reports are one-based.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
import math
from numbers import Real
from pathlib import Path
import random
import shutil
import unicodedata
from uuid import uuid4


def normalize_text(text: str) -> str:
    """Preserve case and accents while normalizing NFC and whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFC", " ".join(text.split()))


def normalized_source(text: str) -> str:
    """Return the NFC, whitespace-cleaned, case-insensitive leakage key."""
    return unicodedata.normalize("NFC", normalize_text(text).casefold())


@dataclass(frozen=True)
class Pair:
    """An immutable aligned source/target row; construction does not alter text."""

    source: str
    target: str


def read_parallel(sourcepath: str | Path, targetpath: str | Path) -> list[Pair]:
    """Read strict UTF-8 rows, rejecting blank rows and unequal/empty corpora.

    Errors include the one-based physical row. A normal final newline is not a
    blank row; an additional empty line is. No rows are silently discarded.
    """
    pairs = []
    sentinel = object()
    with Path(sourcepath).open(encoding="utf-8") as source, Path(targetpath).open(
        encoding="utf-8"
    ) as target:
        for row, (src, tgt) in enumerate(zip_longest(source, target, fillvalue=sentinel), 1):
            if src is sentinel or tgt is sentinel:
                raise ValueError(f"Unequal parallel lengths at row {row}: {sourcepath}, {targetpath}")
            src, tgt = normalize_text(src), normalize_text(tgt)
            if not src or not tgt:
                side = "source" if not src else "target"
                raise ValueError(f"Blank {side} at row {row}: {sourcepath}, {targetpath}")
            pairs.append(Pair(src, tgt))
    if not pairs:
        raise ValueError("Parallel corpus is empty")
    return pairs


def _clean_pair(pair: Pair) -> Pair:
    if not isinstance(pair, Pair):
        raise TypeError("split rows must be Pair instances")
    result = Pair(normalize_text(pair.source), normalize_text(pair.target))
    if not result.source or not result.target:
        raise ValueError("Split rows must have nonblank source and target")
    return result


def audit_splits(splits: dict[str, list[Pair]]) -> dict:
    """Return a JSON-serializable audit, without modifying the supplied rows.

    ``counts`` maps split names to row counts. ``duplicate_sources`` maps each
    split to normalized source keys with repeated one-based row indices.
    ``duplicate_pairs`` maps splits to source/target/rows records (exact after
    NFC/whitespace cleanup, case-sensitive). ``source_overlaps`` maps each
    cross-split source key to split names and their row indices. Variants with
    distinct targets are allowed within a split, but never across splits.
    """
    counts, duplicate_sources, duplicate_pairs = {}, {}, {}
    locations = defaultdict(dict)
    for name, pairs in splits.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Split names must be nonempty strings")
        sources, exact = defaultdict(list), defaultdict(list)
        counts[name] = len(pairs)
        for row, pair in enumerate(pairs, 1):
            pair = _clean_pair(pair)
            sources[normalized_source(pair.source)].append(row)
            exact[pair].append(row)
        duplicate_sources[name] = {key: rows for key, rows in sources.items() if len(rows) > 1}
        duplicate_pairs[name] = [
            {"source": pair.source, "target": pair.target, "rows": rows}
            for pair, rows in exact.items() if len(rows) > 1
        ]
        for key, rows in sources.items():
            locations[key][name] = rows
    return {
        "counts": counts,
        "duplicate_sources": duplicate_sources,
        "duplicate_pairs": duplicate_pairs,
        "source_overlaps": {key: places for key, places in locations.items() if len(places) > 1},
    }


def validate_splits(splits: dict[str, list[Pair]]) -> dict:
    """Return the audit or raise ValueError for leakage/exact duplicate pairs."""
    report = audit_splits(splits)
    problems = []
    if report["source_overlaps"]:
        problems.append(f"cross-split source leakage ({len(report['source_overlaps'])} groups)")
    for name, duplicates in report["duplicate_pairs"].items():
        if duplicates:
            problems.append(f"duplicate exact pairs in {name} ({len(duplicates)} groups)")
    if problems:
        raise ValueError("; ".join(problems))
    return report


def _file_lineage(path: str | Path, include_hashes: bool) -> dict:
    path = Path(path)
    result = {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
    if include_hashes:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def corpus_lineage(source: str | Path, target: str | Path, *, include_hashes: bool = True) -> dict:
    """Describe input paths/byte sizes and optionally streaming SHA-256 hashes."""
    return {"source": _file_lineage(source, include_hashes),
            "target": _file_lineage(target, include_hashes)}


def prepare_splits(
    source: str | Path,
    target: str | Path,
    outputdir: str | Path,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict:
    """Write grouped splits and manifest.json; return the manifest dictionary.

    Fractions apply to unique normalized-source groups, not rows. Held-out
    group counts are floored, with the remainder assigned to training. A
    positive held-out fraction that yields zero groups fails before writing;
    zero fractions deliberately produce empty held-out files. Exact cleaned
    pairs are deduplicated, preserving case/target variants and first occurrence order.

    Files are named {train,validation,test}.{source,target}.txt. The manifest
    records input and output hashes, seed, policy, counts, and removed rows.
    A sibling staging directory is renamed into place only when complete.
    Nonempty outputs are never overwritten. Directory publication assumes no
    concurrent writer; failure cleanup is confined to this call's staging dir.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    for name, fraction in (("validation_fraction", validation_fraction), ("test_fraction", test_fraction)):
        if isinstance(fraction, bool) or not isinstance(fraction, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError(f"{name} must be finite and in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    outputdir = Path(outputdir)
    if outputdir.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    if outputdir.exists() and (not outputdir.is_dir() or any(outputdir.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {outputdir}")
    lineage = corpus_lineage(source, target)
    pairs = read_parallel(source, target)
    if lineage != corpus_lineage(source, target):
        raise ValueError("Input corpus changed while being read")
    unique = list(dict.fromkeys(pairs))
    groups = defaultdict(list)
    for pair in unique:
        groups[normalized_source(pair.source)].append(pair)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_validation = math.floor(len(keys) * validation_fraction)
    n_test = math.floor(len(keys) * test_fraction)
    for name, fraction, count in (
        ("validation", validation_fraction, n_validation),
        ("test", test_fraction, n_test),
    ):
        if fraction > 0 and count == 0:
            raise ValueError(
                f"Requested {name} split would be empty: {len(keys)} normalized-source "
                f"groups with fraction {fraction}; increase the fraction or corpus size, "
                "or set the fraction to zero to disable this partition"
            )
    selected = {
        "train": keys[n_validation + n_test:],
        "validation": keys[:n_validation],
        "test": keys[n_validation:n_validation + n_test],
    }
    splits = {name: [pair for key in subset for pair in groups[key]] for name, subset in selected.items()}
    audit = validate_splits(splits)
    manifest = {
        "schema_version": 1,
        "inputs": lineage,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "normalization": "NFC; collapse Unicode whitespace; preserve case and accents",
        "grouping": "NFC casefolded source; sorted keys; seeded shuffle; floor held-out group counts",
        "input_rows": len(pairs),
        "unique_pairs": len(unique),
        "duplicates_removed": len(pairs) - len(unique),
        "blank_rows_dropped": 0,
        "source_groups": len(groups),
        "counts": audit["counts"],
        "splits": {},
    }
    outputdir.parent.mkdir(parents=True, exist_ok=True)
    staging = outputdir.parent / f".{outputdir.name}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        for name, rows in splits.items():
            entry = {"rows": len(rows), "source_groups": len(selected[name])}
            for side in ("source", "target"):
                filename = f"{name}.{side}.txt"
                path = staging / filename
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    for pair in rows:
                        stream.write(getattr(pair, side) + "\n")
                metadata = _file_lineage(path, True)
                metadata["path"] = filename
                entry[side] = metadata
            manifest["splits"][name] = entry
        with (staging / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        if outputdir.exists():
            outputdir.rmdir()  # Fails if another writer has populated it.
        staging.rename(outputdir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
