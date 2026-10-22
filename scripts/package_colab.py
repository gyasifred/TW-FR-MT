"""Create a source-only Colab upload bundle, excluding weights and credentials."""

from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / "TW-FR-MT-colab.zip"
    output.parent.mkdir(exist_ok=True)
    roots = ["twfr_mt", "configs", "notebooks", "docs", "scripts", "tests", "legacy", ".github", "TW_FR_EN_corpus/data"]
    files = [root / name for name in ["README.md", "LICENSE", "pyproject.toml", "requirements.txt", "app.py", "Dockerfile", ".dockerignore", ".gitignore"]]
    for directory in roots:
        files.extend(path for path in (root / directory).rglob("*") if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path.suffix in {".bin", ".pt", ".pth", ".safetensors", ".pyc"} or path.name == ".env":
                continue
            archive.write(path, "TW-FR-MT/" + path.relative_to(root).as_posix())
    print(output)


if __name__ == "__main__":
    main()
