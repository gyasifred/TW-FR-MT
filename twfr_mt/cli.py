"""Explicit entry points. Importing the package never starts training."""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path


def serve(registry, model_root, device, host, port, share, allow_public):
    user, password = os.environ.get("TWFR_USERNAME"), os.environ.get("TWFR_PASSWORD")
    if bool(user) != bool(password):
        raise ValueError("Set both TWFR_USERNAME and TWFR_PASSWORD, or neither")
    if (share or host not in {"127.0.0.1", "localhost", "::1"}) and not user and not allow_public:
        raise ValueError("Remote serving requires authentication or explicit --allow-public consent")
    from .ui import create_app

    app = create_app(registry, model_root, device=device)
    app.launch(server_name=host, server_port=port, share=share,
               auth=(user, password) if user else None, show_error=False,
               blocked_paths=[str(Path(model_root).resolve())],
               max_file_size="1mb", enable_monitoring=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="twfr-mt", description="Train, evaluate and serve translation checkpoints")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source", required=True)
    preparation.add_argument("--target", required=True)
    preparation.add_argument("--output-dir", required=True)
    preparation.add_argument("--seed", type=int, default=42)
    preparation.add_argument("--validation-fraction", type=float, default=0.1)
    preparation.add_argument("--test-fraction", type=float, default=0.1)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--references", required=True)
    evaluation.add_argument("--output")
    translation = commands.add_parser("translate")
    translation.add_argument("--registry", required=True)
    translation.add_argument("--model-root", required=True)
    translation.add_argument("--route", required=True)
    inputs = translation.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--input-file")
    translation.add_argument("--output-file")
    translation.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    conversion = commands.add_parser("export-checkpoint")
    conversion.add_argument("--source", required=True)
    conversion.add_argument("--destination", required=True)
    conversion.add_argument("--allow-legacy-bin", action="store_true")
    server = commands.add_parser("serve")
    server.add_argument("--registry", default=os.environ.get("TWFR_REGISTRY", "configs/models.drive.json"))
    server.add_argument("--model-root", default=os.environ.get("TWFR_MODEL_ROOT", "/content/drive/MyDrive/MT"))
    server.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    server.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    server.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    server.add_argument("--share", action="store_true")
    server.add_argument("--allow-public", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from .training import TrainingConfig, run_training
            values = json.loads(Path(args.config).read_text(encoding="utf-8"))
            unknown = set(values) - {field.name for field in fields(TrainingConfig)}
            if unknown:
                raise ValueError(f"Unknown training fields: {sorted(unknown)}")
            print(json.dumps(run_training(TrainingConfig(**values)), indent=2, default=str))
        elif args.command == "prepare":
            from .data import prepare_splits
            print(json.dumps(prepare_splits(args.source, args.target, args.output_dir, seed=args.seed, validation_fraction=args.validation_fraction, test_fraction=args.test_fraction), indent=2, default=str))
        elif args.command == "evaluate":
            from .metrics import score_translations
            result = score_translations(Path(args.predictions).read_text(encoding="utf-8-sig").splitlines(), Path(args.references).read_text(encoding="utf-8-sig").splitlines())
            text = json.dumps(result, indent=2)
            if args.output:
                Path(args.output).write_text(text + "\n", encoding="utf-8")
            print(text)
        elif args.command == "translate":
            from .config import load_routes
            from .inference import TranslationService
            translator = TranslationService(load_routes(args.registry, args.model_root), args.model_root, device=args.device)
            texts = [args.text] if args.text is not None else Path(args.input_file).read_text(encoding="utf-8-sig").splitlines()
            if not texts or any(not text.strip() for text in texts):
                raise ValueError("Translation inputs cannot contain empty rows")
            translated = [translator.translate(text, args.route) for text in texts]
            if args.output_file:
                output = Path(args.output_file)
                if output.exists():
                    raise ValueError("Refusing to overwrite an existing translation output")
                output.write_text("\n".join(translated) + "\n", encoding="utf-8")
            else:
                print("\n".join(translated))
        elif args.command == "export-checkpoint":
            from .export import export_checkpoint
            print(export_checkpoint(args.source, args.destination, allow_legacy_bin=args.allow_legacy_bin))
        else:
            serve(args.registry, args.model_root, args.device, args.host, args.port, args.share, args.allow_public)
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
