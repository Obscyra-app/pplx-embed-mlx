# Native MLX retrieval and full text fine-tuning on a $0.93 GPU lease

I built a native Apple Silicon runtime for [Perplexity's pplx-embed-v2-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b), then compared a small adapter, full text fine-tuning and a six-layer distilled student. The trained models improved retrieval on a held-out banking task. The student also showed a substantial cross-domain regression, so I am releasing it as a task-specific research checkpoint.

[Runtime and evidence](https://github.com/Obscyra-app/pplx-embed-mlx) · [Original MLX FP16](https://huggingface.co/Nurymanau/pplx-embed-v2-late-0.6b-MLX-fp16) · [BANKING full](https://huggingface.co/Nurymanau/pplx-embed-v2-late-0.6b-BANKING-MLX) · [BANKING student](https://huggingface.co/Nurymanau/pplx-embed-v2-late-BANKING-Student6-MLX)

This is independent community work, without affiliation or endorsement by Perplexity.

## What runs locally

The runtime produces token-level embeddings and ranks documents with MaxSim. It includes native MLX inference, a tokenizer and processor, and an index/search CLI. Torch and Transformers are not inference dependencies. This is a custom runtime, not a drop-in `mlx_lm` or Sentence Transformers checkpoint.

The original FP16 model is the general starting point. It supports text queries and text or bounded image documents; image validation remains limited. Upstream weights are pinned to revision `8fc2de24534aa3610d85fa59c463313a5f096455`. The published packages include licenses, SHA256 manifests and provenance; repackaging did not change weight bytes.

## A small, controlled training experiment

The task uses [PolyAI BANKING77](https://huggingface.co/datasets/PolyAI/banking77) as intent-exemplar retrieval. It is not the official classification benchmark, and these scores are not answer accuracy.

Training used 1,232 queries and 154 exemplars. Dev used 308 queries and 154 documents, kept separate from training. A fresh test contained 770 queries, 154 documents and 77 intents. Exact and near matches with earlier experimental tests were excluded. Checkpoints were selected on dev before final test evaluation.

The full model trained all 493.86 million text parameters. The student trained 374.14 million parameters across six retained text layers. Each arm used two seeds, learning rates of 1e-5 and 3e-5, and three epochs: 24 epochs across eight runs. Both selected recipes used 3e-5 at epoch three. Vision weights were excluded from training; the student omits vision entirely.

## What improved

These results were measured with native MLX FP16 on the same fresh BANKING test:

| Model | nDCG@10 | Hit@1 | Recall@10 |
|---|---:|---:|---:|
| Original base | 74.05% | 70.91% | 84.87% |
| Base + small adapter | 77.74% | 74.55% | 87.86% |
| Full text fine-tuning | **89.49%** | **87.01%** | **95.26%** |
| Six-layer student | 81.79% | 78.83% | 91.30% |

Full training exceeded the adapted teacher by 11.75 percentage points in nDCG@10; the paired bootstrap over 77 intent clusters gave a 95% interval of [+9.20, +14.53]. The student gained 4.05 points, with an interval of [+1.53, +6.67]. Independent executable checks verified selection, metrics and exported tensors.

The improvement belongs to the complete training recipe. There was no matched head-only control under the same recipe, so this experiment does not isolate the causal benefit of unfreezing internal layers.

## What did not transfer

On NanoSciFact, evaluated in FP32 CUDA with 40 queries over 2,919 documents, nDCG@10 was 85.81% for the adapted teacher, 84.62% for full training and 66.05% for the student. Full-model Recall@10 stayed at 92.5%, but Hit@1 fell from 80% to 75%. The student lost 19.75 nDCG points relative to the teacher.

That is why the original FP16 model remains the general default. The student is a BANKING-specific experiment. We did not evaluate the initial student on NanoSciFact, so its final gap does not separate compression effects from training effects. Full native MLX Nano evaluation of the newly trained models was not run, and image quality after full text fine-tuning is unvalidated.

## Size, latency and cost

The student weight file is 748.6 MB versus 1,189.3 MB for the full model. It has 24.2% fewer text parameters; the larger 37.1% file reduction also reflects removal of vision.

On an M3 Mac with 16 GB memory, eight fixed dev inputs gave warm medians of 26.16 ms for the student and 48.04 ms for the adapted teacher, about 1.84×. This used two warmups and seven measured passes. It is a small latency fixture, not an application throughput claim.

One RTX 2000 Ada with 16 GB VRAM handled the GPU phase. The lease lasted 13,875 seconds at $0.24/hour: an estimated **$0.925** for the training, evaluation and rescue lease. This is a time/rate estimate rather than an invoice; storage, R2 and local Mac costs are excluded. The GPU was terminated after verified checkpoint rescue.

## Try it

On Apple Silicon, clone the [runtime repository](https://github.com/Obscyra-app/pplx-embed-mlx), install its requirements, and run `python download_weights.py --variant base-fp16`. The downloader pins an immutable HF revision and verifies checksums. The README contains a two-document index/search example; each standalone HF card also includes direct download instructions.

The [complete bundle](https://huggingface.co/Nurymanau/pplx-embed-v2-late-0.6b-MLX) preserves the adapter, optional mixed MLP-only Q4 conversion, and historical failed student alongside the successful BANKING variants. Q4 was not faster in our measurements. Exact results and caveats are in [P10_VALIDATION.json](https://github.com/Obscyra-app/pplx-embed-mlx/blob/main/P10_VALIDATION.json) and the [experiment report](https://github.com/Obscyra-app/pplx-embed-mlx/blob/main/P10_README.md).

Weights follow the upstream MIT declaration; runtime code is Apache-2.0. BANKING77 is CC BY 4.0, from Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807). No raw dataset is redistributed.
