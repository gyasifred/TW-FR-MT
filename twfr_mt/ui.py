"""Gradio translation conversation UI; chat history is not model context."""

from pathlib import Path

from .config import load_routes
from .inference import TranslationService


def create_app(registry_path: str | Path, model_root: str | Path, device: str = "auto", service: TranslationService | None = None):
    import gradio as gr

    routes = load_routes(registry_path, model_root)
    translator = service or TranslationService(routes, model_root, device=device)
    choices = [(route.label, route.id) for route in routes.values()]
    initial = choices[0][1]

    def status(route_id):
        return "Checkpoint found" if translator.readiness(route_id) else "Awaiting checkpoint"

    def respond(message, history, route_id, output_length, beams):
        del history
        try:
            return translator.translate(message, route_id, int(output_length), int(beams))
        except FileNotFoundError:
            raise gr.Error("Checkpoint unavailable for this route.") from None
        except ValueError as exc:
            raise gr.Error(str(exc)) from None
        except (RuntimeError, OSError):
            raise gr.Error("Translation unavailable. Ask the operator to check the checkpoint and runtime.") from None

    with gr.Blocks(title="Twi-French Translation", analytics_enabled=False, fill_height=True) as app:
        gr.Markdown("# Twi–French")
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                route = gr.Dropdown(choices=choices, value=initial, label="Translation route")
                checkpoint_status = gr.Textbox(value=status(initial), label="Model status", interactive=False)
                output_length = gr.Slider(16, 256, value=128, step=16, label="Maximum output tokens")
                beams = gr.Slider(1, 8, value=4, step=1, label="Beam count")
                route.change(status, route, checkpoint_status, api_name=False)
            with gr.Column(scale=4, min_width=280):
                gr.ChatInterface(
                    respond,
                    chatbot=gr.Chatbot(label="Translations", height=480, render_markdown=False, buttons=["copy"]),
                    textbox=gr.Textbox(placeholder="Twi / French text", lines=3, max_lines=8),
                    additional_inputs=[route, output_length, beams],
                    flagging_mode="never", save_history=False, analytics_enabled=False,
                    concurrency_limit=1, api_visibility="private",
                )
    app.queue(max_size=16, default_concurrency_limit=1)
    return app
