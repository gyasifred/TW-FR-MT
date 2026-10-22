"""One-time conversion of trusted local training exports to safetensors."""

from pathlib import Path


def export_checkpoint(source, destination, allow_legacy_bin=False):
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("Source must be a local Hugging Face checkpoint directory")
    if destination.exists():
        raise ValueError("Export destination must not exist")
    safe = (source / "model.safetensors").is_file() or (source / "model.safetensors.index.json").is_file()
    if not safe and not allow_legacy_bin:
        raise ValueError("Legacy .bin conversion requires --allow-legacy-bin and a trusted source")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(source), local_files_only=True, trust_remote_code=False, use_safetensors=safe, weights_only=True)
    model.save_pretrained(str(destination), safe_serialization=True)
    tokenizer.save_pretrained(str(destination))
    return destination
