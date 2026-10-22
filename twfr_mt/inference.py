"""Bounded, cached local-checkpoint inference without import-time model downloads."""

from collections import OrderedDict
from pathlib import Path
import json
import threading
from typing import Callable, Protocol
import unicodedata

from .config import TranslationRoute, TranslationStep, checkpoint_path


class Translator(Protocol):
    def translate(self, text: str, step: TranslationStep, max_new_tokens: int, num_beams: int) -> str: ...


class LocalSeq2SeqTranslator:
    def __init__(self, path: Path, device: str = "auto", max_source_tokens: int = 256):
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"No Hugging Face checkpoint at {path}")
        if not ((path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()):
            raise ValueError("Only safetensors checkpoints are accepted; export a trusted legacy model first")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the inference extra: pip install -e '.[inference]'") from exc
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to(self.device).eval()
        if not self.model.config.is_encoder_decoder:
            raise ValueError("A sequence-to-sequence translation checkpoint is required")
        self.language_defaults = {attribute: getattr(self.tokenizer, attribute, None) for attribute in ("src_lang", "tgt_lang")}
        metadata = path / "twfr_metadata.json"
        self.metadata = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
        model_limit = getattr(self.model.config, "max_position_embeddings", max_source_tokens)
        self.max_source_tokens = min(max_source_tokens, model_limit)

    def translate(self, text: str, step: TranslationStep, max_new_tokens: int, num_beams: int) -> str:
        for attribute in ("source_lang", "target_lang", "source_prefix", "source_tokenizer_lang", "target_tokenizer_lang"):
            if attribute in self.metadata and self.metadata[attribute] != getattr(step, attribute):
                raise ValueError("Translation route conflicts with the checkpoint's saved language configuration")
        for attribute, value in (("src_lang", step.source_tokenizer_lang), ("tgt_lang", step.target_tokenizer_lang)):
            if hasattr(self.tokenizer, attribute):
                if value is not None and value not in self.tokenizer.get_vocab():
                    raise ValueError("Configured language token is absent from the tokenizer vocabulary")
                setattr(self.tokenizer, attribute, value if value is not None else self.language_defaults[attribute])
            elif value is not None:
                raise ValueError(f"This tokenizer does not support {attribute}")
        options = {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": False}
        if step.target_tokenizer_lang:
            vocabulary = self.tokenizer.get_vocab()
            if step.target_tokenizer_lang not in vocabulary:
                raise ValueError("Target language token is absent from the tokenizer vocabulary")
            options["forced_bos_token_id"] = vocabulary[step.target_tokenizer_lang]
        encoded = self.tokenizer(step.source_prefix + text, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > self.max_source_tokens:
            raise ValueError(f"Input exceeds this checkpoint's {self.max_source_tokens}-token limit")
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, **options)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


class TranslationService:
    """Serialize shared model/tokenizer access and bound resident checkpoints."""

    def __init__(self, routes: dict[str, TranslationRoute], model_root: str | Path, device: str = "auto", loader: Callable[[Path], Translator] | None = None, max_cached_models: int = 2):
        if max_cached_models < 1:
            raise ValueError("Model cache must have a positive capacity")
        self.routes = routes
        self.model_root = Path(model_root).expanduser().resolve()
        self.loader = loader or (lambda path: LocalSeq2SeqTranslator(path, device=device))
        self.cache: OrderedDict[Path, Translator] = OrderedDict()
        self.capacity = max_cached_models
        self.lock = threading.RLock()

    def readiness(self, route_id: str) -> bool:
        for step in self.routes[route_id].steps:
            path = checkpoint_path(self.model_root, step.checkpoint)
            if not (path / "config.json").is_file() or not ((path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()):
                return False
        return True

    def translate(self, text: str, route_id: str, max_new_tokens: int = 128, num_beams: int = 4) -> str:
        if route_id not in self.routes:
            raise ValueError("Unknown translation route")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Translation requires nonempty text")
        if len(text) > 4000:
            raise ValueError("Input must contain at most 4000 characters")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError("Maximum output length must be between 1 and 256 tokens")
        if type(num_beams) is not int or not 1 <= num_beams <= 8:
            raise ValueError("Beam count must be between 1 and 8")
        translated = unicodedata.normalize("NFC", text).strip()
        with self.lock:
            for step in self.routes[route_id].steps:
                path = checkpoint_path(self.model_root, step.checkpoint)
                if path not in self.cache:
                    if len(self.cache) >= self.capacity:
                        self.cache.popitem(last=False)
                    try:
                        self.cache[path] = self.loader(path)
                    except (OSError, ValueError, RuntimeError) as exc:
                        raise RuntimeError("The configured checkpoint could not be loaded") from exc
                self.cache.move_to_end(path)
                translated = self.cache[path].translate(translated, step, max_new_tokens, num_beams)
                if not isinstance(translated, str) or not translated.strip():
                    raise ValueError("Checkpoint produced no translation")
        return translated
