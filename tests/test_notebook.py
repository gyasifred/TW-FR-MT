"""Check the configured notebook route without accessing Google Drive."""

import ast
import json
from pathlib import Path

import pytest

from twfr_mt.config import load_routes


@pytest.mark.parametrize("missing", [None, "val_tw", "checkpoint"])
def test_inventory_actual_paths(tmp_path, missing):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/00_drive_inventory.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = next("".join(cell["source"]) for cell in cells
                  if cell["cell_type"] == "code" and "DATA_FILES =" in "".join(cell["source"]))
    tree = ast.parse(source)
    roots = {"DATA_ROOT": "/content/drive/MyDrive/data",
             "DRIVE_ROOT": "/content/drive/MyDrive/MT"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in roots:
                assert isinstance(node.value, ast.Call)
                assert node.value.args[0].value == roots[name]
                node.value.args[0] = ast.Constant(str(tmp_path / name))
    data = tmp_path / "DATA_ROOT"
    models = tmp_path / "DRIVE_ROOT"
    data.mkdir()
    models.mkdir()
    if missing != "checkpoint":
        (models / "OPUS-mt-tw-fr-tuned").mkdir()
    filenames = {"train_en", "train_twi", "train_fr", "val_en", "val_tw",
                 "val_fr", "test_en", "test_twi", "test_fr"}
    for filename in filenames - {missing}:
        (data / filename).touch()
    namespace = {}
    code = compile(ast.fix_missing_locations(tree), str(notebook), "exec")
    if missing:
        expected = "OPUS-mt-tw-fr-tuned" if missing == "checkpoint" else missing
        with pytest.raises(FileNotFoundError, match=expected):
            exec(code, namespace)
    else:
        exec(code, namespace)
        assert {path.name for split in namespace["DATA_FILES"].values()
                for path in split.values()} == filenames
        assert namespace["CHECKPOINT_PATH"] == models / "OPUS-mt-tw-fr-tuned"


@pytest.mark.parametrize("weights", ["model.safetensors", "pytorch_model.bin", None])
def test_drive_model_configuration(tmp_path, weights):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/02_colab_translate.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = next("".join(cell["source"]) for cell in cells
                  if cell["cell_type"] == "code" and "CHECKPOINT_RELATIVE =" in "".join(cell["source"]))
    tree = ast.parse(source)
    registry = tmp_path / "registries/tw-fr.json"
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"DRIVE_ROOT", "REGISTRY_PATH"}:
                path = tmp_path if name == "DRIVE_ROOT" else registry
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                      args=[ast.Constant(str(path))], keywords=[])
    checkpoint = tmp_path / "OPUS-mt-tw-fr-tuned"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (checkpoint / weights).touch()
    code = compile(ast.fix_missing_locations(tree), str(notebook), "exec")
    if weights == "pytorch_model.bin":
        with pytest.raises(ValueError, match="Legacy .bin weights found"):
            exec(code, {"Path": Path})
        assert not registry.exists()
    elif weights is None:
        with pytest.raises(FileNotFoundError, match="model weights"):
            exec(code, {"Path": Path})
        assert not registry.exists()
    else:
        exec(code, {"Path": Path})
        route = load_routes(registry, tmp_path)["tw-fr"]
        assert (route.source_lang, route.target_lang) == ("tw", "fr")
        assert route.steps[0].checkpoint == "OPUS-mt-tw-fr-tuned"
