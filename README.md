# Perplexity late-0.6B on MLX

Unofficial native Apple Silicon inference plus BANKING retrieval adaptation and full text fine-tuning. **The original FP16 model is the general default. Trained variants are task-specific research checkpoints.** No affiliation or endorsement by Perplexity; no first-port claim.

[Code on GitHub](https://github.com/Obscyra-app/pplx-embed-mlx) · [Weights on Hugging Face](https://huggingface.co/Nurymanau/pplx-embed-v2-late-0.6b-MLX) · [Detailed P10 results](P10_README.md)

Upstream: [perplexity-ai/pplx-embed-v2-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b), pinned revision `8fc2de24534aa3610d85fa59c463313a5f096455`. Token-level embeddings use MaxSim retrieval. This is a standalone custom MLX runtime, **not a drop-in `mlx_lm` or Sentence Transformers checkpoint**. It imports neither Torch nor Transformers for inference.

## Choose a model

- [Original Perplexity late-0.6B — native MLX FP16](https://huggingface.co/Nurymanau/pplx-embed-v2-late-0.6b-MLX-fp16)
- [Perplexity late-0.6B — BANKING full text fine-tuning on MLX](https://huggingface.co/Nurymanau/pplx-embed-v2-late-0.6b-BANKING-MLX)
- [Perplexity late — six-layer BANKING student on MLX](https://huggingface.co/Nurymanau/pplx-embed-v2-late-BANKING-Student6-MLX)

The original bundle remains available, including the small adapter, mixed MLP-Q4 and historical research variants. Each separate model page contains the native runtime and its own root-level weights.

[Technical write-up](https://github.com/Obscyra-app/pplx-embed-mlx/blob/main/docs/experiment.md) · [Apple Silicon MLX portfolio](https://huggingface.co/collections/Nurymanau/apple-silicon-mlx-ports-and-experiments-6ac9709d0628f31e0993a8f7)

## Measured P10 result

Native MLX FP16 on the same fresh BANKING intent-exemplar retrieval test: 770 queries,154 documents,77 intents. This is not the official classification benchmark or answer accuracy.

| Variant | nDCG@10 | Hit@1 | Recall@10 |
|---|---:|---:|---:|
| Original base | 74.05% | 70.91% | 84.87% |
| Base + small adapter | 77.74% | 74.55% | 87.86% |
| Full text fine-tuning | **89.49%** | **87.01%** | **95.26%** |
| Six-layer P10 student | 81.79% | 78.83% | 91.30% |

Full vs adapted teacher: +11.75 percentage points nDCG, paired77-intent bootstrap95%CI[+9.20,+14.53]. Two seeds × two learning rates × three epochs for each arm; checkpoints selected only on separate dev data. All493.86M text parameters trained in the full model,374.14M in the student. This tests the complete recipe; there is no matched head-only ablation isolating the contribution of internal layers.

**Cross-domain limitation:** NanoSciFact FP32 CUDA nDCG is85.81% for the adapted teacher,84.62% for full training and66.05% for the student. Full Recall stays92.5%, but Hit@1 falls80→75%. Student is BANKING-specific and is not a generic substitute. Nano uses40queries over2919documents; full native MLX Nano evaluation of the newly trained models was not run. Image quality after full text fine-tuning is unvalidated. See [all limits](P10_README.md).

Student file748.6MB vs1189.3MB full FP16;24.2% fewer text parameters. File reduction also reflects omission of vision. Warm medians26.16ms vs48.04ms teacher on8fixeddev inputs on an M3/16GB Mac (~1.84×); not application throughput or a general speed guarantee. GPU training/eval/rescue lease cost estimate$0.925 at$0.24/hour, excluding storage/R2, not an invoice.

## Included variants

| Folder under `weights/` | Role | Approximate weight file |
|---|---|---:|
| `base-fp16` | Original general default; text and bounded image documents | 1.19GB |
| `base-mlp-q4` | Optional mixed MLP-only Q4; not the entire model in4bit | 999MB |
| `banking-adapter` | 65.6KB adapter; requires `base-fp16` | 65.6KB |
| `banking-full-p10-fp16` | Full text fine-tuning, BANKING research | 1.19GB |
| `banking-student6-p10` | Six-layer BANKING student; large Nano regression | 749MB |
| `student6-experimental` | Historical P8 output-distilled student; FAILED quality gate | 749MB |

Mixed MLP-Q4 retained NanoSciFact nDCG84.34→84.08% and Recall92.5% on the original model. It is not advertised as faster. Earlier P7/P8 results use a different616-query test and must not be mixed with the P10 table: see [historical evidence](P1_P9_RESULTS.md).

## Quick start

Apple Silicon required. Tested Python3.12, MLX0.32.3, macOS26.6.2. From the GitHub checkout:

```sh
git clone https://github.com/Obscyra-app/pplx-embed-mlx.git
cd pplx-embed-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_weights.py --variant base-fp16
```

The downloader uses an immutable HF revision and verifies SHA256. Alternatively download this entire HF snapshot, including `code/`, and run the following commands from its root; the downloader step is only needed for a GitHub checkout.

Create `documents.json`:

```json
[{"id":"train","text":"The train leaves at 08:30."},{"id":"refund","text":"A refund can take several business days."}]
```

```sh
PYTHONPATH=code python -m pplx_mlx.cli --model weights/base-fp16 index --documents documents.json --output my-index
PYTHONPATH=code python -m pplx_mlx.cli --model weights/base-fp16 search --index my-index --query "When does the train leave?" --top-k 2
```

For trained full weights download `--variant banking-full-p10-fp16` and replace the model path. For the P10 student use `banking-student6-p10`. These variants already contain their learned projection: **do not add the P6 adapter**. Always rebuild the index when changing variants.

For the small adapter download both `base-fp16` and `banking-adapter`, then add `--adapter weights/banking-adapter` to both index and search. For Q4 use `--model weights/base-mlp-q4 --bits 4`. The historical P8 student uses `--model weights/student6-experimental`.

Original FP16 supports text queries and text/image documents: use `{"id":"page","image":"relative/page.png"}` in the documents array. Trained variants are validated for text only; students have no vision encoder. One unpadded input per call; query≤1024 and document≤4096tokens, oversized inputs rejected. Image resize has explicit bounded limits. Avoid parallel model jobs on small-memory Macs.

## Evidence and licenses

`P10_VALIDATION.json` has exact metrics, both seeds, confidence intervals, gates, sizes, timing and cost. `P10_VERIFICATION.json` is a pre-publication local package check; its historical `published:false` describes that check's time. `VALIDATION.json` covers earlier stages. Raw datasets, training infrastructure, private history, optimizer states and secrets are not distributed. Upstream pretraining contamination is unknown.

Weights: upstream MIT declaration, see `WEIGHTS_LICENSE.txt`. Runtime code: Apache-2.0 (`LICENSE`), with third-party notices in `code/pplx_mlx/NOTICE` and `licenses/`. BANKING77: PolyAI, CC BY4.0, Casanueva et al.(2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807); dataset license in `DATA_LICENSE.txt`. No dataset redistribution here.

`SHA256SUMS.json` covers every HF payload file except itself. GitHub has a separate source manifest and an independent root history; model weights remain on HF.
