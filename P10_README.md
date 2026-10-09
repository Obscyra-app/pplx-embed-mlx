# P10: full text-backbone training and six-layer distillation

Release v0.1.0. Independent community experiment; no affiliation with Perplexity.

All text embeddings, layers, normalization and projection were trainable: 493,859,776 parameters in the full model and 374,135,776 in the student. Vision was excluded from training; image quality of the full trained variant has not been validated.

The student has about 24.2% fewer text parameters. Total-file comparisons also include the full model's vision encoder, which the student omits; file reduction is not the same as text-parameter reduction.

Fixed grid: two seeds, two learning rates, three epochs; selection used only dev. Fresh BANKING intent-exemplar retrieval test: 770 queries and 154 documents. This is not the official classification benchmark. Previous tests were not training or selection data.

| Native MLX FP16 | nDCG@10 | Hit@1 | Recall@10 |
|---|---:|---:|---:|
| base | 74.05% | 70.91% | 84.87% |
| teacher | 77.74% | 74.55% | 87.86% |
| full | 89.49% | 87.01% | 95.26% |
| student | 81.79% | 78.83% | 91.30% |

- full: PASS frozen acceptance gates; weights/banking-full-p10-fp16.
- student: PASS frozen acceptance gates; weights/banking-student6-p10.

| NanoSciFact, FP32 CUDA | nDCG@10 | Hit@1 | Recall@10 |
|---|---:|---:|---:|
| base | 84.34% | 75.00% | 92.50% |
| teacher | 85.81% | 80.00% | 92.50% |
| full | 84.62% | 75.00% | 92.50% |
| student | 66.05% | 47.50% | 82.50% |

The student's acceptance gates cover BANKING retrieval, file size and latency; NanoSciFact is a reporting-only regression check. P10 did not measure NanoSciFact for the initial P8 student, so the student/teacher gap does not isolate layer removal from training effects. The full model's Nano gates cover nDCG and Recall, so a passing gate does not imply unchanged Hit@1. The student has a substantial cross-domain regression: NanoSciFact nDCG falls by 19.75 percentage points versus the adapted teacher. Treat it as a BANKING-specific research checkpoint.

The original FP16 model remains the generic default. Trained variants are task-specific; a failed gate remains a documented research result. NanoSciFact regression was measured on FP32 CUDA; this does not imply a native full-corpus Nano result. MLX parity includes two long regression documents; latency covers eight fixed dev inputs with two warmups and seven measurements.

Use --model weights/banking-full-p10-fp16 or --model weights/banking-student6-p10 without an additional adapter. Rebuild the retrieval index. Use these trained variants for text; the student has no vision encoder. The standalone runtime requires no Torch or Transformers.

Exact gates, both seed results, confidence intervals, measured sizes, warm timings and GPU cost estimate are in P10_VALIDATION.json. Cost is a time/rate estimate, not an invoice; storage and R2 are separate. The experiment tests the full training recipe and does not isolate the contribution of unfreezing the backbone from learning the projection.

Upstream model: perplexity-ai/pplx-embed-v2-late-0.6b, revision 8fc2de24534aa3610d85fa59c463313a5f096455. Dataset: PolyAI BANKING77, CC BY 4.0; Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders, https://arxiv.org/abs/2003.04807. Existing model/code licenses and DATA_LICENSE.txt apply; no raw training or evaluation data are shipped.
