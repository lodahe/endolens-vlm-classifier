"""Evaluate the same inference path used by the app."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from qwen_archive import (
    CHECKPOINT, REVIEW_THRESHOLD, ROOT, SPLIT_HASH, QwenArchive,
    apply_review, frozen_rows, select_threshold,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int, help="Run only the first N images for a smoke test.")
    parser.add_argument("--calibrate-review", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_output")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.calibrate_review and (args.split != "validation" or args.limit is not None):
        parser.error("Calibration requires the full validation split.")

    rows = [
        row for row in frozen_rows("split_manifest.csv", SPLIT_HASH)
        if row["status"] == "included" and row["split"] == args.split
    ][:args.limit]
    from PIL import Image

    with ZipFile(ROOT / "data/hyper-kvasir-labeled-images.zip") as archive:
        def load_sample(row):
            contents = archive.read(row["archive_member"])
            if hashlib.sha256(contents).hexdigest() != row["sha256"]:
                raise ValueError(f"Image checksum mismatch: {row['archive_member']}")
            with Image.open(BytesIO(contents)) as image:
                return image.copy()

        model = QwenArchive()
        model.predict(load_sample(rows[0]))  # One warm-up; excluded from latency.
        torch = model.torch
        torch.cuda.reset_peak_memory_stats(model.device)
        predictions = []
        for index, row in enumerate(rows, 1):
            predictions.append(model.predict(load_sample(row)))
            if index % 100 == 0 or index == len(rows):
                print(f"{index}/{len(rows)}", flush=True)
        peak = torch.cuda.max_memory_allocated(model.device) / (1024 ** 2)

    threshold = REVIEW_THRESHOLD
    if args.calibrate_review:
        threshold = select_threshold(zip((row["label"] for row in rows), predictions))
    output = args.output / (args.split if args.limit is None else f"{args.split}_limit_{args.limit}")
    output.mkdir(parents=True, exist_ok=True)
    paired = []
    confusion = Counter()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "truth", "label", "raw_output", "score",
                         "decision", "review_reason", "latency_ms"])
        for row, pred in zip(rows, predictions, strict=True):
            truth = row["label"]
            decision, reason = apply_review(pred["label"], pred["score"], True, threshold)
            label = pred["label"] or "INVALID"
            paired.append((truth, label, decision))
            confusion[truth, label] += 1
            writer.writerow([row["archive_member"], truth, label, pred["raw_output"],
                             pred["score"], decision, reason, pred["latency_ms"]])

    with (output / "confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = (*model.labels, "INVALID")
        writer.writerow(["truth / prediction", *columns])
        for label in model.labels:
            writer.writerow([label, *(confusion[label, candidate] for candidate in columns)])

    def metrics(group):
        total = len(group)
        accepted = sum(decision != "REVIEW" for _, _, decision in group)
        correct = sum(truth == label for truth, label, _ in group)
        accepted_correct = sum(truth == decision for truth, _, decision in group)
        return dict(
            n=total, correct=correct, accepted=accepted, accepted_correct=accepted_correct,
            top1_accuracy=correct / total if total else None,
            review_rate=(total - accepted) / total if total else None,
            accepted_accuracy=accepted_correct / accepted if accepted else None,
        )

    summary = dict(
        checkpoint=CHECKPOINT, split=args.split, threshold_source=(
            "full_validation" if args.calibrate_review else "notebook_12_validation"
        ),
        review_threshold=threshold if threshold != float("inf") else None,
        review_all=threshold == float("inf"), gpu=torch.cuda.get_device_name(model.device),
        batch_size=1, warmup_images=1, peak_allocated_mib=peak,
        median_latency_ms=median(pred["latency_ms"] for pred in predictions),
        overall=metrics(paired),
        per_class={label: metrics([row for row in paired if row[0] == label])
                   for label in model.labels},
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
