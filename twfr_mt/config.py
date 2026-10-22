"""Validated, operator-managed translation routes and local checkpoint boundaries."""

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath


LANGUAGES = {"tw": "Twi", "fr": "French", "en": "English"}


@dataclass(frozen=True)
class TranslationStep:
    source_lang: str
    target_lang: str
    checkpoint: str
    source_tokenizer_lang: str | None = None
    target_tokenizer_lang: str | None = None
    source_prefix: str = ""


@dataclass(frozen=True)
class TranslationRoute:
    id: str
    label: str
    source_lang: str
    target_lang: str
    steps: tuple[TranslationStep, ...]


def checkpoint_path(model_root: str | Path, checkpoint: str) -> Path:
    root = Path(model_root).expanduser().resolve()
    if not isinstance(checkpoint, str):
        raise ValueError("Checkpoint names must be strings")
    relative = Path(checkpoint.replace("\\", "/"))
    if not checkpoint or relative.is_absolute() or relative.drive or PureWindowsPath(checkpoint).drive:
        raise ValueError("Checkpoint names must be relative to the configured model root")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Checkpoint must stay inside the configured model root")
    return resolved


def load_routes(registry_path: str | Path, model_root: str | Path) -> dict[str, TranslationRoute]:
    raw = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"routes"} or not isinstance(raw["routes"], list):
        raise ValueError("Registry must contain only a routes array")
    routes = {}
    for entry in raw["routes"]:
        if not isinstance(entry, dict):
            raise ValueError("Each route must be an object")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in routes:
            raise ValueError("Route IDs must be unique nonempty strings")
        try:
            steps = tuple(TranslationStep(**step) for step in entry["steps"])
            fields = {key: value for key, value in entry.items() if key != "steps"}
            route = TranslationRoute(**fields, steps=steps)
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid translation route schema") from exc
        if not isinstance(route.label, str) or not route.label.strip() or not 1 <= len(steps) <= 2:
            raise ValueError("Routes require a label and one direct or two pivot steps")
        chain = route.source_lang
        for step in steps:
            if step.source_lang != chain or step.source_lang not in LANGUAGES or step.target_lang not in LANGUAGES:
                raise ValueError("Route steps must form a supported continuous language chain")
            if step.source_lang == step.target_lang:
                raise ValueError("Translation steps must change language")
            checkpoint_path(model_root, step.checkpoint)
            if not isinstance(step.source_prefix, str):
                raise ValueError("Source prefix must be text")
            for token in (step.source_tokenizer_lang, step.target_tokenizer_lang):
                if token is not None and (not isinstance(token, str) or not token):
                    raise ValueError("Tokenizer language identifiers must be nonempty strings")
            chain = step.target_lang
        if chain != route.target_lang or route.source_lang == route.target_lang:
            raise ValueError("Route endpoints must match its steps")
        routes[identifier] = route
    if not routes:
        raise ValueError("Configure at least one translation route")
    return routes
