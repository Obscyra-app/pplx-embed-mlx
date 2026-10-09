# Historical P1–P9 candidate record

This is preserved pre-publication evidence. The current release and P10 results are in README.md.

# Perplexity late-0.6B: unofficial native MLX release candidate

P10 adds full text fine-tuning and a trained six-layer student. See [the new results, trained weights and cross-domain limitations](P10_README.md). The original FP16 model remains the generic default. The earlier results below describe P1–P9. This package is still a local review candidate.

Local candidate; not yet uploaded. Upstream model: perplexity-ai/pplx-embed-v2-late-0.6b,
revision 8fc2de24534aa3610d85fa59c463313a5f096455. Upstream model card declares MIT.
Independent community port; no affiliation or endorsement; no first-port claim.

Validated default: original FP16, optionally with the separate BANKING adapter.
The six-layer student FAILED its quality gate and is supplied for research only.

| Measured retrieval test | nDCG@10 | Recall@10 |
|---|---:|---:|
| Fresh BANKING, base | 76.51% | 87.26% |
| Fresh BANKING, base + adapter | 79.54% | 89.85% |
| Fresh BANKING, distilled six-layer student | 64.49% | 77.84% |
| NanoSciFact, FP16 | 84.34% | 92.50% |
| NanoSciFact, MLP-only Q4 | 84.08% | 92.50% |

Fresh BANKING is 616 held-out queries over 154 intent exemplars. Adapter selection
used separate P6 dev data; the adapter was frozen before this fresh test. Paired
77-intent bootstrap: nDCG delta +3.03 percentage points, 95% CI [+1.04,+5.05].
NanoSciFact is 40 held-out queries over 2919 documents, reused for regression.
Student weights are 748.6 MB vs 1189.3 MB FP16. Four-short-input warm medians
were 19.9 vs 41.6 ms on M3; this limited microbenchmark does not offset its
15.05-point nDCG deficit to the adapted teacher. Stages P1–P9 used no cloud GPU; P10 usage is reported separately.

Requires Apple Silicon and the pinned packages in requirements.txt. Neural inference
imports neither Torch nor Transformers. Run from this directory:

```sh
pip install -r requirements.txt
PYTHONPATH=code python -m pplx_mlx.cli --model weights/base-fp16 index --documents documents.json --output my-index
PYTHONPATH=code python -m pplx_mlx.cli --model weights/base-fp16 search --index my-index --query "Find the relevant document"
```

documents.json is an array of {"id":"unique-id","text":"document text"} or
{"id":"unique-id","image":"relative/image.png"}. FP16 supports text queries and
text/image documents. Single unpadded input per call; query<=1024 and document<=4096
tokens; oversized inputs rejected. Images use a bounded local resize policy.

For optional mixed quantization use --model weights/base-mlp-q4 --bits 4.
Only language MLP matrices are Q4; the complete model is NOT four-bit.
This mode is not advertised as a speedup. See VALIDATION.json for full-corpus quality.

For the experimental BANKING77 adapter add --adapter weights/banking-adapter
to both index and search commands with base-fp16. Rebuild the index. Text only;
other precision/base or mismatched query/index adapters are rejected. The added
weight matrix is 65,616 file bytes; the full encoder is still required. Data:
PolyAI BANKING77 CC BY 4.0; Casanueva et al. (2020), Efficient Intent Detection
with Dual Sentence Encoders, https://arxiv.org/abs/2003.04807. See DATA_LICENSE.txt.

The student6-experimental folder is a research checkpoint retaining six inherited
text layers with output-only token/ranking distillation. It has no vision encoder.
Use --model weights/student6-experimental without --adapter or --bits. Its quality
gate and limitations are in VALIDATION.json; it is NOT the recommended default.

Evaluation is small and task-specific. BANKING relevance means matching intent,
not answer correctness; these are custom exemplar-retrieval tests, not the official
classification benchmark. NanoSciFact has only 40 held-out queries. Pretraining
contamination is unknown. Four-input warm latency is a microbenchmark, not a general
speed guarantee. Preserve known failures when reporting results. All training and
evaluation data are excluded from this candidate.

Third-party licenses and attributions are in licenses/ and code/pplx_mlx/NOTICE.
The proposed code license for this unpublished candidate is Apache-2.0 (LICENSE);
upstream model weights remain under their declared MIT terms. DATA_LICENSE.txt
records the BANKING77 dataset license; no dataset is redistributed here.
Weights, packed tensor schemas and tokenizer files have SHA manifests; preserve them.
The local experiment used pacing, timeouts and swap guards. Avoid parallel model
jobs on small-memory Macs. SHA256SUMS.json covers every shipped file except itself.


## P10 full-training addendum

See [P10 results and trained variants](P10_README.md). Publication remains on hold.
