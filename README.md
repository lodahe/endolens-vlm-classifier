# EndoLens

A local browser demo for categorizing gastrointestinal endoscopy images into 23 HyperKvasir archive labels, with optional human review and JSON export. It uses Qwen2.5-VL-3B-Instruct with the LoRA adapter selected for Notebook 12's REVIEW evaluation and Notebook 13's final-model comparison.

## Run on Windows

Use Python 3.11 and a CUDA-capable NVIDIA GPU. The existing `vlm` environment uses PyTorch `2.11.0+cu128`. From the repository root:

```powershell
conda activate vlm
python -m pip install -r requirements-app.txt
python app.py
```

For a fresh environment, install the matching CUDA build of PyTorch first, then install `requirements-model.txt` and `requirements-app.txt`.

Open http://127.0.0.1:7860 or the address printed by Gradio. Upload a JPG or PNG and click **Classify image**. The first request loads the weights. Later requests reuse the model. When the optional HyperKvasir ZIP is present, you can also load a validation example by class.

- **REVIEW** is on by default. Low-scoring, invalid, and unscored responses go to REVIEW. Unchecking the option applies the change to the current output without running the model again. An invalid label is displayed as INVALID with REVIEW disabled.
- For a REVIEW result, optionally choose a human label and export it alongside the original model output as JSON. A blank human label leaves the decision pending.
- The score is the mean log-probability of generated non-special tokens. Notebook 12 selected the published threshold of `-0.0176234495` on validation. The stored value is rounded to the precision printed by the notebook.
- Inference latency includes image preprocessing, generation, and scoring, with CUDA synchronization. Upload, queue wait, display, and first-request model loading are excluded. Model-load time is reported separately when applicable.

## Local dependencies

| Path relative to repo root | Purpose |
|---|---|
| `models/Qwen2.5-VL-3B-Instruct/` | Complete local base model and processor files |
| `runs/qwen_archive_23_lora_v1/epoch_04/` | `adapter_config.json` and `adapter_model.safetensors` from the trained run |
| `data/hyperkvasir_23_v1/rare_labels.csv` | Frozen category order, generated locally by Notebook 11 |
| `data/hyperkvasir_23_v1/split_manifest.csv` | Frozen split and image checksums, generated locally by Notebook 10 |
| `data/hyper-kvasir-labeled-images.zip` | Original HyperKvasir labeled-image archive, needed for evaluation and optional validation examples |

The first public release contains source code and documentation. The dataset, split CSVs, base model, and LoRA checkpoint are required locally. Rebuilding the exact frozen split also requires the original pilot manifests under `data/hyperkvasir_lora/` and `data/hyperkvasir_lora_expanded/` to preserve prior-use exclusions. These manifests and the fine-tuned checkpoint are not bundled. Notebook 12 documents the training and validation workflow. Its existing cells use `E:/VLM`; adjust the root in the notebook if you work from another location.

## Reproduce the held-out evaluation

```powershell
python evaluate.py --split test --limit 5
python evaluate.py --split test
python evaluate.py --split validation --calibrate-review
```

The full test uses the frozen 1,594-image split. Evaluation writes per-image predictions, the raw Top-1 confusion matrix, and overall and per-class metrics to `evaluation_output/`. The reported CUDA peak is allocated GPU memory including model weights. Latency excludes archive I/O and uses one warm-up image, with batch size 1. Validation recalibration reports the exact threshold independently. The test command always uses the frozen Notebook 12 threshold. The app uses greedy generation like Notebook 12. Notebook 13 uses candidate ranking for its separate Top-2 experiment.

## Source release

The public release includes Notebooks 10–12 for split construction, rare-label analysis, and LoRA training and REVIEW evaluation. Notebook outputs are cleared in the public copies. Early CLIP baselines, animal-image experiments, pilot notebooks, model weights, images, and the running Notebook 13 experiment are excluded from this package.
