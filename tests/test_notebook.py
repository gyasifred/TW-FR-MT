"""Check the configured notebook route without accessing Google Drive."""

import ast
import json
from pathlib import Path

import pytest

from twfr_mt.config import load_routes


ROOT = Path(__file__).resolve().parents[1]


def notebook_code(filename, marker):
    cells = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))["cells"]
    return next("".join(cell["source"]) for cell in cells
                if cell["cell_type"] == "code" and marker in "".join(cell["source"]))


def test_maintained_paths_and_configs():
    for notebook in (ROOT / "notebooks").glob("*.ipynb"):
        document = json.loads(notebook.read_text(encoding="utf-8"))
        text = json.dumps(document)
        for stale in ("CHOOSE_DATA_SHORTCUT_PATH", "/MyDrive/TW-FR-MT", "source_en.txt", "target_tw.txt"):
            assert stale not in text
        for cell in document["cells"]:
            if cell["cell_type"] == "code":
                # IPython's package-install magic is not Python syntax.
                source = "".join(line for line in cell["source"] if not line.startswith("%"))
                ast.parse(source)
    config = json.loads((ROOT / "configs/train.example.json").read_text(encoding="utf-8"))
    for name in ("train_source", "train_target", "validation_source", "validation_target"):
        assert config[name].startswith("/content/drive/MyDrive/MT/prepared_data/en-tw-run-001/")
    assert config["output_dir"] == "/content/drive/MyDrive/MT/runs/en-tw-run-001"
    route = load_routes(ROOT / "configs/models.drive.json", ROOT / "models")["tw-fr"]
    assert route.steps[0].checkpoint == "OPUS-mt-tw-fr-tuned"


@pytest.mark.parametrize("problem", [None, "overlap", "unequal", "changed_saved"])
def test_training_notebook_preserves_actual_splits(tmp_path, problem):
    data, models = tmp_path / "data", tmp_path / "MT"
    data.mkdir()
    models.mkdir()
    contents = {
        "train_en": "Hello\nThank you\n", "train_twi": "Agoo\nMedaase\n",
        "val_en": "Welcome\n", "val_tw": "Akwaaba\n",
        "test_en": "Goodbye\n", "test_twi": "Nante yiye\n",
    }
    if problem == "overlap":
        contents["test_en"] = "Hello\n"
    if problem == "unequal":
        contents["val_tw"] += "Extra\n"
    for name, content in contents.items():
        (data / name).write_text(content, encoding="utf-8")
    tree = ast.parse(notebook_code("01_colab_train.ipynb", "DATA_FOLDER_ID ="))
    replacements = {"DATA_ROOT": data, "DRIVE_ROOT": models,
                    "SPLIT_DIR": tmp_path / "splits", "LOCAL_DATA_DIR": tmp_path / "inputs"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                      args=[ast.Constant(str(replacements[name]))], keywords=[])
    namespace = {"Path": Path}
    exec(compile(ast.fix_missing_locations(tree), "configuration", "exec"), namespace)
    preparation = notebook_code("01_colab_train.ipynb", "from twfr_mt.data import corpus_lineage")
    if problem == "changed_saved":
        saved = models / "prepared_data/en-tw-run-001"
        saved.mkdir(parents=True)
        (saved / "manifest.json").write_text("{}", encoding="utf-8")
    if problem:
        message = {"overlap": "cross-split source leakage", "unequal": "Unequal parallel lengths",
                   "changed_saved": "Saved split provenance differs"}[problem]
        with pytest.raises(ValueError, match=message):
            exec(preparation, namespace)
    else:
        exec(preparation, namespace)
        saved = models / "prepared_data/en-tw-run-001"
        assert namespace["audit"]["counts"] == {"train": 2, "validation": 1, "test": 1}
        for split, names in {"train": ("train_en", "train_twi"),
                             "validation": ("val_en", "val_tw"),
                             "test": ("test_en", "test_twi")}.items():
            for side, name in zip(("source", "target"), names):
                assert (saved / f"{split}.{side}.txt").read_text(encoding="utf-8") == contents[name]
        manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["inputs"]["validation"]["target"]["path"] == str((data / "val_tw").resolve())
    for name, content in contents.items():
        assert (data / name).read_text(encoding="utf-8") == content


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
