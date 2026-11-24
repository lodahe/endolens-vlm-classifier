"""Local inference shared by the browser demo and frozen-split evaluation."""
import csv
import hashlib
import math
from pathlib import Path
from statistics import mean
from time import perf_counter


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/hyperkvasir_23_v1"
MODEL = ROOT / "models/Qwen2.5-VL-3B-Instruct"
ADAPTER = ROOT / "runs/qwen_archive_23_lora_v1/epoch_04"
CHECKPOINT = "qwen_archive_23_lora_v1/epoch_04"
REVIEW_THRESHOLD = -0.0176234495  # Printed by Notebook 12 on validation.
LABELS_HASH = "609fc9ddfacd0db2584e6410158eaa59a7396c85036f4005d5bc82e6475d1688"
SPLIT_HASH = "11f36db99e3762145a3974c5fb57772b0120adf48cb353ee1e486cce9920c633"


def frozen_rows(name, digest):
    path = DATA / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Frozen metadata checksum mismatch: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_labels():
    labels = tuple(row["label"] for row in frozen_rows("rare_labels.csv", LABELS_HASH))
    if len(labels) != 23 or len(set(labels)) != 23:
        raise ValueError("Expected 23 unique archive labels.")
    return labels


def apply_review(label, score, enabled, threshold=REVIEW_THRESHOLD):
    if label is None:
        return ("REVIEW", "Output did not match any archive label.") if enabled else ("INVALID", None)
    if enabled and score is None:
        return "REVIEW", "No finite token score."
    if enabled and score < threshold:
        return "REVIEW", "Token score is below the validation threshold."
    return label, None


def select_threshold(rows, target=0.97):
    eligible = sorted(
        ((truth, prediction) for truth, prediction in rows
         if prediction["label"] is not None and prediction["score"] is not None),
        key=lambda row: row[1]["score"],
        reverse=True,
    )
    correct = 0
    threshold = math.inf
    for count, (truth, prediction) in enumerate(eligible, 1):
        correct += truth == prediction["label"]
        score = prediction["score"]
        if (count == len(eligible) or eligible[count][1]["score"] != score) and correct / count >= target:
            threshold = score
    return threshold


class QwenArchive:
    def __init__(self):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        for path in (MODEL / "config.json", ADAPTER / "adapter_config.json",
                     ADAPTER / "adapter_model.safetensors"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing local model file: {path}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Launch the app in the vlm environment.")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.labels = load_labels()
        self.prompt = (
            "Archive this gastrointestinal endoscopy image. Reply with one exact label from: "
            + ", ".join(self.labels) + "."
        )
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        self.processor.image_processor.size = {"shortest_edge": 56 * 56, "longest_edge": 448 * 448}
        self.special_ids = set(self.processor.tokenizer.all_special_ids)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, local_files_only=True,
        ).to(self.device)
        self.model = PeftModel.from_pretrained(base, ADAPTER).eval()
        torch.cuda.synchronize(self.device)

    def predict(self, image):
        from PIL import ImageOps

        torch = self.torch
        torch.cuda.synchronize(self.device)
        start = perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt},
        ]}]
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
            output = self.model.generate(
                **inputs, max_new_tokens=16, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
            tokens = output.sequences[0, inputs.input_ids.shape[1]:]
            raw = self.processor.batch_decode(tokens.unsqueeze(0), skip_special_tokens=True)[0]
            answer = raw.strip().lower().removesuffix(".")
            candidate = answer.split("(", 1)[0].strip() if answer.endswith(")") else answer
            log_probs = [
                logits[0].float().log_softmax(-1)[token].item()
                for token, logits in zip(tokens.tolist(), output.scores, strict=True)
                if token not in self.special_ids
            ]
        score = mean(log_probs) if log_probs else -math.inf
        torch.cuda.synchronize(self.device)
        return {
            "label": candidate if candidate in self.labels else None,
            "raw_output": raw,
            "score": score if math.isfinite(score) else None,
            "latency_ms": (perf_counter() - start) * 1000,
        }
