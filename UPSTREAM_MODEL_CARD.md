---
license: mit
pipeline_tag: feature-extraction
tags:
- feature-extraction
- sentence-similarity
- mteb
- sentence-transformers
language:
- multilingual
---

<p align="center">
  <img src="assets/logo.svg" alt="Perplexity Logo" width="400">
</p>

<p align="center">pplx-embed-v2-late: Multimodal Late-Interaction Embeddings</p>

`pplx-embed-v2-late` is a family of multimodal late-interaction (ColBERT) retrievers for text, images, and visual documents, built on Qwen3.5 with bidirectional attention. The models produce one 128-dimensional vector per token and score query–document similarity using MaxSim. The 0.6B and 9B models share an embedding space, allowing the 0.6B model to query an index built with the 9B model.

For benchmarks and details, see our [blog post](https://www.perplexity.ai/hub/blog/multimodal-embeddings-beyond-a-single-vector).

## Models

| Model | Active Params | Dimension | Public ViDoRe(v3), Image, nDCG@10 | Public ViDoRe(v3), Markdown, nDCG@10 |
|:-----:|:-------------:|:---------:|:------------------:|:----------------------------------:|
| [`pplx-embed-v2-late-0.6b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b) | 340M | 128 | 62.3% | 61.2% |
| [`pplx-embed-v2-late-9b`](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-9b) | 7.4B | 128 | 65.2% | 64.7% |

## Training

Both models were distilled from an internal 18B ColBERT teacher trained on pair and triplet data. Distillation used a token-level [LEAF-style](https://aclanthology.org/2026.acl-long.2008/) objective. The 0.6B model was fully fine-tuned. For the 9B model, the final eight transformer layers were fully fine-tuned, while the remaining transformer layers and vision encoder were adapted with LoRA.

## Usage (Sentence Transformers)

Requires **sentence-transformers >= 6.0.0** and **transformers >= 5.4.0**.

```bash
pip install 'sentence-transformers>=6.0.0' 'transformers>=5.4.0'
```

```python
from PIL import Image
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder(
    "perplexity-ai/pplx-embed-v2-late-0.6b",  # or "perplexity-ai/pplx-embed-v2-late-9b"
    device="cuda",
)
queries = model.encode_query(["what statute governs limitations?"])
documents = model.encode_document(["A document passage"])
scores = model.similarity(queries, documents)  # MaxSim

image = Image.open("document.png").convert("RGB")
image_documents = model.encode_document([image])
image_scores = model.similarity(queries, image_documents)
```

The export uses native Sentence Transformers modules and needs no custom Python
code.
Use separate encoding calls for text-only and image-only batches. Mixed text+image inputs are not supported.

PyLate inserts Q/D markers at the second position; this model expects them first.

## Citation

```bibtex
@online{perplexity2026MultimodalEmbeddingsBeyondASingleVector,
  author = {{Perplexity} and Markus Krimmel and Louis Milliken and Antoine Chaffin and Sarah Eslami and Ekaterina Glazkova and Maximilian Schall and Bo Wang and Denis Bykov and Jarrod Bryan and Claire Beamer and Jeremy Yang},
  title = {{Multimodal embeddings beyond a single vector}},
  date = {2026-10-07},
  url = {https://www.perplexity.ai/hub/blog/multimodal-embeddings-beyond-a-single-vector}
}
```
