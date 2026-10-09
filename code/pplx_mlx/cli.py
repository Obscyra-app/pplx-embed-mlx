"""Local late-interaction retrieval: python -m experiments.pplx_mlx.cli."""

import argparse
import json
import time
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .processor import Processor


def maxsim(query, document):
    """Sum of per-query-token maxima, as in the official Perplexity runtime."""
    if query.ndim != 2 or document.ndim != 2 or not len(query) or not len(document):
        raise ValueError("Expected nonempty token matrices")
    return float((query @ document.T).max(axis=1).sum())


def fingerprint(path):
    h = hashlib.sha256()
    if (Path(path) / "student.json").exists():
        names = (
            "student.json",
            "student.safetensors",
            "tokenizer.json",
            "config.json",
            "2_MultiVectorMask/config.json",
        )
    elif (Path(path) / "mlx_checkpoint.json").exists():
        names = (
            "mlx_checkpoint.json",
            "mlx_model.safetensors",
            "tokenizer.json",
            "config.json",
            "2_MultiVectorMask/config.json",
        )
    else:
        names = (
            "model.safetensors",
            "1_Dense/model.safetensors",
            "tokenizer.json",
            "config.json",
            "2_MultiVectorMask/config.json",
        )
    for name in names:
        h.update(name.encode())
        with (Path(path) / name).open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/pplx-late-0.6b/source")
    p.add_argument(
        "--dtype", choices=["float16", "float32", "bfloat16"], default="float16"
    )
    p.add_argument("--bits", type=int, choices=[4, 6, 8])
    p.add_argument(
        "--adapter", help="Optional pinned token-adapter directory; text only"
    )
    sub = p.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument(
        "--documents",
        required=True,
        help="JSON array of {id,text} or {id,image}; image paths relative to JSON",
    )
    index.add_argument("--output", required=True)
    search = sub.add_parser("search")
    search.add_argument("--index", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    a = p.parse_args()
    from .model import Encoder
    import mlx.core as mx

    identity = fingerprint(a.model)
    from .adapter import TokenAdapter, check_index_adapter

    adapter = TokenAdapter(a.adapter, identity, a.dtype, a.bits) if a.adapter else None
    processor = Processor(a.model)
    if (Path(a.model) / "student.json").exists():
        if a.dtype != "float16" or a.bits is not None:
            raise ValueError("Experimental student requires FP16 without quantization")
        from .next.student import Student

        model = Student(a.model)
    else:
        model = Encoder(a.model, a.dtype, a.bits)

    def encode(value, task):
        if adapter and not isinstance(value, str):
            raise ValueError("The experimental adapter is validated for text only")
        batch = processor(value, task)
        out = model(**batch)
        vectors = adapter(out["embeddings"]) if adapter else out["embeddings"]
        return np.asarray(vectors)[batch["output_mask"]]

    start = time.perf_counter()
    if a.command == "index":
        src = Path(a.documents)
        docs = json.loads(src.read_text())
        ids = [d["id"] for d in docs]
        if not docs or len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique; index must be nonempty")
        dest = Path(a.output)
        dest.mkdir(parents=True, exist_ok=False)
        vectors = {}
        for i, doc in enumerate(docs):
            if ("text" in doc) == ("image" in doc):
                raise ValueError("Each document must have exactly one of text/image")
            value = (
                doc["text"]
                if "text" in doc
                else Image.open(src.parent / doc["image"]).convert("RGB")
            )
            vectors[f"d{i}"] = encode(value, "document")
        np.savez(dest / "vectors.npz", **vectors)
        (dest / "manifest.json").write_text(
            json.dumps(
                {
                    "documents": docs,
                    "dtype": a.dtype,
                    "bits": a.bits,
                    "source_fingerprint": identity,
                    "adapter_sha256": adapter.identity if adapter else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(
            json.dumps(
                {
                    "documents": len(docs),
                    "seconds": time.perf_counter() - start,
                    "mlx_peak_bytes": mx.get_peak_memory(),
                }
            )
        )
    else:
        root = Path(a.index)
        manifest = json.loads((root / "manifest.json").read_text())
        check_index_adapter(manifest, adapter)
        if manifest["source_fingerprint"] != identity:
            raise ValueError("Index was encoded with different model/tokenizer files")
        if (manifest["dtype"], manifest["bits"]) != (a.dtype, a.bits):
            raise ValueError("Query and index precision differ; specify matching flags")
        docs = np.load(root / "vectors.npz", allow_pickle=False)
        q = encode(a.query, "query")
        hits = [
            {"id": d["id"], "score": maxsim(q, docs[f"d{i}"])}
            for i, d in enumerate(manifest["documents"])
        ]
        hits.sort(key=lambda x: x["score"], reverse=True)
        print(
            json.dumps(
                {
                    "query": a.query,
                    "hits": hits[: a.top_k],
                    "seconds": time.perf_counter() - start,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
