"""Dependency-light IO tests plus read-only original-corpus regressions."""

import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from twfr_mt.data import (
    Pair, audit_splits, corpus_lineage, normalize_text, normalized_source,
    prepare_splits, read_parallel, validate_splits,
)


@pytest.fixture
def tmp_path():
    # Avoid tempfile's private-mode mkdir, denied by some managed Windows ACLs.
    directory = Path(__file__).resolve().parents[1] / f".data-test-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def parallel_files(directory, source, target):
    src, tgt = directory / "source.txt", directory / "target.txt"
    src.write_bytes(source.encode("utf-8"))
    tgt.write_bytes(target.encode("utf-8"))
    return src, tgt


def test_normalization_preserves_case_accents_and_punctuation():
    assert normalize_text(" \t E\u0301cole\u00a0\u2003A!\r\n") == "\u00c9cole A!"
    assert normalized_source(" E\u0301COLE\tA! ") == "\u00e9cole a!"
    assert normalized_source("Stra\u00dfe") == normalized_source("STRASSE")
    assert normalized_source("cafe") != normalized_source("caf\u00e9")
    assert Pair(" A ", " B ").source == " A "
    with pytest.raises(TypeError):
        normalize_text(None)


def test_read_parallel_cleans_utf8_and_keeps_last_row(tmp_path):
    paths = parallel_files(tmp_path, " E\u0301cole\tICI\r\n\u0254ba", " \u00c0 la maison \r\nChild\n")
    assert read_parallel(*paths) == [Pair("\u00c9cole ICI", "\u00c0 la maison"), Pair("\u0254ba", "Child")]


@pytest.mark.parametrize("source,target,row", [
    ("a\nb\n", "a\n", 2), ("a\n", "a\nb\n", 2),
    ("", "a\n", 1), ("a\n", "", 1),
])
def test_read_parallel_rejects_unequal_lengths(tmp_path, source, target, row):
    with pytest.raises(ValueError, match=f"Unequal.*row {row}"):
        read_parallel(*parallel_files(tmp_path, source, target))


@pytest.mark.parametrize("source,target", [
    ("a\n\n", "b\nc\n"), ("a\n", " \t\u00a0\n"), ("\n", "\n"),
])
def test_read_parallel_rejects_blank_rows(tmp_path, source, target):
    with pytest.raises(ValueError, match="Blank"):
        read_parallel(*parallel_files(tmp_path, source, target))


def test_read_parallel_empty_missing_invalid_utf8(tmp_path):
    src, tgt = parallel_files(tmp_path, "", "")
    with pytest.raises(ValueError, match="empty"):
        read_parallel(src, tgt)
    src.write_bytes(b"\xff\n")
    with pytest.raises(UnicodeDecodeError):
        read_parallel(src, tgt)
    with pytest.raises(FileNotFoundError):
        read_parallel(tmp_path / "missing", tgt)


def test_audit_and_validation_report_variants_duplicates_and_leakage():
    splits = {
        "train": [Pair("\u00c9COLE", "School"), Pair(" E\u0301COLE ", "School"), Pair("\u00e9cole", "Academy")],
        "validation": [Pair("\u00e9cole", "Different")],
        "test": [Pair("\u00c9cole", "School")],
    }
    original = {name: list(rows) for name, rows in splits.items()}
    report = audit_splits(splits)
    assert report["counts"] == {"train": 3, "validation": 1, "test": 1}
    assert report["duplicate_sources"]["train"] == {"\u00e9cole": [1, 2, 3]}
    assert report["duplicate_pairs"]["train"] == [{"source": "\u00c9COLE", "target": "School", "rows": [1, 2]}]
    assert report["source_overlaps"] == {"\u00e9cole": {"train": [1, 2, 3], "validation": [1], "test": [1]}}
    json.dumps(report)
    assert splits == original
    with pytest.raises(ValueError, match="cross-split source leakage.*duplicate exact pairs"):
        validate_splits(splits)


def test_validation_allows_variants_but_not_duplicates():
    variants = {"train": [Pair("A", "one"), Pair("a", "two")], "test": []}
    assert not validate_splits(variants)["source_overlaps"]
    with pytest.raises(ValueError, match="duplicate exact pairs"):
        validate_splits({"train": [Pair("A", "one"), Pair("A", "one")]})
    with pytest.raises(ValueError, match="leakage"):
        validate_splits({"train": [Pair("A", "one")], "test": [Pair(" a ", "two")]})
    with pytest.raises(ValueError, match="nonblank"):
        audit_splits({"train": [Pair(" ", "one")]})
    with pytest.raises(TypeError, match="Pair"):
        audit_splits({"train": [("a", "b")]})
    with pytest.raises(ValueError, match="names"):
        audit_splits({"": []})


def test_lineage_hashes_original_bytes_and_can_skip_hashes(tmp_path):
    src, tgt = parallel_files(tmp_path, " E\u0301 \r\n", "x\n")
    lineage = corpus_lineage(src, tgt)
    assert lineage["source"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert lineage["source"]["size_bytes"] == len(src.read_bytes())
    assert lineage["source"]["path"] == str(src.resolve())
    assert "sha256" not in corpus_lineage(src, tgt, include_hashes=False)["source"]


def test_prepare_grouped_deduplicated_reproducible_and_aligned(tmp_path):
    source = [f"Source {i}" for i in range(20)] + ["Source 0", " SOURCE\t0 ", "Source 0"]
    target = [f"Target {i}" for i in range(20)] + ["Variant", "Case variant", "Target 0"]
    paths = parallel_files(tmp_path, "\n".join(source), "\n".join(target))
    before = [p.read_bytes() for p in paths]
    output = tmp_path / "prepared"
    manifest = prepare_splits(*paths, output, 123, 0.2, 0.2)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["input_rows"] == 23
    assert manifest["unique_pairs"] == 22
    assert manifest["duplicates_removed"] == 1
    assert manifest["blank_rows_dropped"] == 0
    assert manifest["source_groups"] == 20
    assert {name: entry["source_groups"] for name, entry in manifest["splits"].items()} == {"train": 12, "validation": 4, "test": 4}
    splits = {name: read_parallel(output / f"{name}.source.txt", output / f"{name}.target.txt") for name in manifest["splits"]}
    validate_splits(splits)
    combined = [pair for rows in splits.values() for pair in rows]
    assert set(combined) == set(read_parallel(*paths))
    assert sum(len(rows) for rows in splits.values()) == 22
    assert [p.read_bytes() for p in paths] == before
    for name, entry in manifest["splits"].items():
        assert manifest["counts"][name] == len(splits[name])
        for side in ("source", "target"):
            path = output / entry[side]["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry[side]["sha256"]
            assert b"\r" not in path.read_bytes()
    again = tmp_path / "again"
    assert prepare_splits(*paths, again, 123, 0.2, 0.2) == manifest
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {p.name: p.read_bytes() for p in again.iterdir()}
    changed = prepare_splits(*paths, tmp_path / "changed", 321, 0.2, 0.2)
    assert changed["splits"] != manifest["splits"]


def test_prepare_small_corpus_and_empty_output_directory(tmp_path):
    paths = parallel_files(tmp_path, "A\na\n", "one\ntwo\n")
    output = tmp_path / "empty"
    output.mkdir()
    manifest = prepare_splits(*paths, output, 0, 0, 0)
    assert manifest["counts"] == {"train": 2, "validation": 0, "test": 0}
    assert (output / "test.source.txt").read_bytes() == b""
    assert (output / "validation.target.txt").read_bytes() == b""


@pytest.mark.parametrize("validation,test,empty_split", [
    (0.1, 0, "validation"), (0, 0.1, "test"), (0.1, 0.1, "validation"),
])
def test_prepare_rejects_requested_empty_heldout_before_writing(tmp_path, validation, test, empty_split):
    paths = parallel_files(tmp_path, "A\na\n", "one\ntwo\n")
    output = tmp_path / "new-parent" / "prepared"
    with pytest.raises(ValueError, match=f"{empty_split} split would be empty.*1 normalized-source"):
        prepare_splits(*paths, output, 0, validation, test)
    assert not output.parent.exists()


@pytest.mark.parametrize("validation,test,error", [
    (-0.1, 0.1, ValueError), (0.1, 1, ValueError), (0.6, 0.4, ValueError),
    (float("nan"), 0.1, ValueError), (0.1, float("inf"), ValueError),
    ("0.1", 0.1, TypeError), (True, 0.1, TypeError),
])
def test_prepare_rejects_invalid_fractions_before_output(tmp_path, validation, test, error):
    paths = parallel_files(tmp_path, "a", "b")
    output = tmp_path / "prepared"
    with pytest.raises(error):
        prepare_splits(*paths, output, 1, validation, test)
    assert not output.exists()


def test_prepare_no_overwrite_or_partial_output(tmp_path, monkeypatch):
    paths = parallel_files(tmp_path, "a", "b")
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_splits(*paths, output)
    assert marker.read_text(encoding="utf-8") == "keep"
    with pytest.raises(FileExistsError):
        prepare_splits(*paths, marker)
    with pytest.raises(TypeError, match="seed"):
        prepare_splits(*paths, tmp_path / "invalid-seed", True)
    paths[1].write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Blank"):
        prepare_splits(*paths, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    paths[1].write_text("b", encoding="utf-8")
    def fail_dump(*args, **kwargs):
        raise OSError("simulated write failure")
    monkeypatch.setattr("twfr_mt.data.json.dump", fail_dump)
    with pytest.raises(OSError, match="simulated"):
        prepare_splits(*paths, tmp_path / "failed", validation_fraction=0, test_fraction=0)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_prepare_rejects_changed_inputs(tmp_path, monkeypatch):
    paths = parallel_files(tmp_path, "a", "b")
    original = read_parallel
    def changing_reader(*args):
        pairs = original(*args)
        paths[0].write_text("changed", encoding="utf-8")
        return pairs
    monkeypatch.setattr("twfr_mt.data.read_parallel", changing_reader)
    with pytest.raises(ValueError, match="changed"):
        prepare_splits(*paths, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


CORPUS = Path(__file__).resolve().parents[1] / "TW_FR_EN_corpus" / "data"


def test_original_totals_have_aligned_nonblank_rows():
    for language in ("en", "fr", "tw"):
        rows = (CORPUS / "total" / f"total_{language}.txt").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 10708
        assert all(normalize_text(row) for row in rows)
    assert len(read_parallel(CORPUS / "total/total_en.txt", CORPUS / "total/total_tw.txt")) == 10708


def test_prepare_original_english_twi_totals_without_modifying_corpus(tmp_path):
    paths = (CORPUS / "total/total_en.txt", CORPUS / "total/total_tw.txt")
    before = corpus_lineage(*paths)
    output = tmp_path / "prepared"
    manifest = prepare_splits(*paths, output, 42, 0.1, 0.1)
    splits = {name: read_parallel(output / f"{name}.source.txt", output / f"{name}.target.txt") for name in ("train", "validation", "test")}
    assert not validate_splits(splits)["source_overlaps"]
    assert sum(manifest["counts"].values()) == len(set(read_parallel(*paths)))
    assert manifest["input_rows"] == 10708
    assert manifest["inputs"] == before == corpus_lineage(*paths)


def test_original_french_test_alignment_defect_is_rejected():
    assert len((CORPUS / "test/test_fr.txt").read_text(encoding="utf-8").splitlines()) == 1070
    for language in ("tw", "en"):
        assert len((CORPUS / "test" / f"test_{language}.txt").read_text(encoding="utf-8").splitlines()) == 1071
        with pytest.raises(ValueError, match="Unequal.*row 1071"):
            read_parallel(CORPUS / "test" / f"test_{language}.txt", CORPUS / "test/test_fr.txt")


def test_original_twi_source_leakage_is_reported_and_rejected():
    paths = {"train": "training/train_tw.txt", "validation": "validation/val_tw.txt", "test": "test/test_tw.txt"}
    splits = {name: [Pair(row, f"target {i}") for i, row in enumerate((CORPUS / path).read_text(encoding="utf-8").splitlines())] for name, path in paths.items()}
    overlaps = audit_splits(splits)["source_overlaps"]
    for left, right, expected in (("train", "validation", 45), ("train", "test", 40), ("validation", "test", 2)):
        assert sum(left in places and right in places for places in overlaps.values()) == expected
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(splits)
