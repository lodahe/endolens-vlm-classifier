"""Browser interface for local Qwen image archiving."""
import json
from io import BytesIO
import logging
import os
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from uuid import uuid4
from zipfile import ZipFile

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

from qwen_archive import (
    CHECKPOINT, REVIEW_THRESHOLD, ROOT, SPLIT_HASH, QwenArchive,
    apply_review, frozen_rows, load_labels,
)


def build_app():
    labels = load_labels()
    archive_path = ROOT / "data/hyper-kvasir-labeled-images.zip"
    samples = {}
    if archive_path.is_file():
        for row in frozen_rows("split_manifest.csv", SPLIT_HASH):
            if row["status"] == "included" and row["split"] == "validation":
                samples.setdefault(row["label"], row["archive_member"])
    engine = None
    export_dir = Path(mkdtemp(prefix="qwen-archive-"))

    def load_sample(label):
        from PIL import Image

        if label is None:
            return None
        with ZipFile(archive_path) as archive:
            with Image.open(BytesIO(archive.read(samples[label]))) as image:
                return image.copy()

    def run_image(image):
        nonlocal engine
        if image is None:
            return None, "Upload an image to begin."
        load_ms = None
        try:
            if engine is None:
                start = perf_counter()
                engine = QwenArchive()
                load_ms = (perf_counter() - start) * 1000
            result = engine.predict(image)
        except (OSError, ValueError) as error:
            logging.exception("Unable to process image")
            return None, str(error)
        except RuntimeError as error:
            logging.exception("Inference failed")
            if "out of memory" in str(error).lower():
                return None, "CUDA ran out of memory. Release GPU memory and retry."
            return None, str(error)
        return dict(result, model_load_ms=load_ms), ""

    def render(result, enabled, message):
        if result is None:
            return message, None, gr.update(value=None, interactive=False), None
        decision, reason = apply_review(result["label"], result["score"], enabled)
        payload = dict(
            result, checkpoint=CHECKPOINT, decision=decision, review_enabled=enabled,
            review_reason=reason, review_threshold=REVIEW_THRESHOLD,
        )
        display = f"{decision}  ·  {result['latency_ms']:.0f} ms"
        if reason:
            display += f"  ·  {reason}"
        return display, payload, gr.update(value=None, interactive=decision == "REVIEW"), None

    def export(payload, manual_label):
        if not payload:
            return None
        path = export_dir / f"archive-{uuid4().hex}.json"
        data = dict(payload, human_label=manual_label if payload["decision"] == "REVIEW" else None)
        path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        return str(path)

    def clear():
        return None, "", "", None, gr.update(value=None, interactive=False), None

    with gr.Blocks(title="Endoscopy Archive") as demo:
        gr.Markdown("# Endoscopy Archive\nQwen2.5-VL 3B · 23 HyperKvasir categories")
        state = gr.State()
        notice = gr.State("")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", image_mode=None, format="png", sources=["upload"], label="Endoscopy image")
                sample = gr.Dropdown(sorted(samples), label="Load a validation example", visible=bool(samples))
                review = gr.Checkbox(value=True, label="Send uncertain results to REVIEW")
                run = gr.Button("Classify image", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Archive decision", interactive=False)
                details = gr.JSON(label="Model output and timing")
                manual = gr.Dropdown(labels, label="Human review label", interactive=False)
                export_button = gr.Button("Export decision")
                download = gr.File(label="Download JSON", interactive=False)
        display_outputs = [status, details, manual, download]
        run.click(clear, outputs=[state, notice, *display_outputs], queue=False).then(
            run_image, image, [state, notice], concurrency_limit=1,
        ).then(render, [state, review, notice], display_outputs)
        sample.input(load_sample, sample, image)
        image.change(clear, outputs=[state, notice, *display_outputs])
        review.change(render, [state, review, notice], display_outputs)
        manual.change(lambda: None, outputs=download, queue=False)
        export_button.click(export, [details, manual], download)
    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", share=False)
