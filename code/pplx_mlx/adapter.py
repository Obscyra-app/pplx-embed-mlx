"""Optional frozen-backbone token adapter. No Torch dependency at inference."""

import hashlib
import json
from pathlib import Path
import numpy as np


class TokenAdapter:
    def __init__(self, path, base_fingerprint, dtype="float16", bits=None):
        from safetensors.numpy import load_file

        path = Path(path)
        self.metadata = json.loads((path / "adapter.json").read_text())
        m = self.metadata
        if m.get("format") != "pplx-token-adapter-v1" or m.get("dimension") != 128:
            raise ValueError("Unsupported adapter format")
        if (
            m["base_fingerprint"] != base_fingerprint
            or dtype != "float16"
            or bits is not None
        ):
            raise ValueError("Adapter requires its pinned unquantized FP16 base")
        data = (path / "adapter.safetensors").read_bytes()
        if hashlib.sha256(data).hexdigest() != m["weights_sha256"]:
            raise ValueError("Adapter checksum mismatch")
        tensors = load_file(str(path / "adapter.safetensors"))
        if set(tensors) != {"weight"}:
            raise ValueError("Unexpected adapter tensors")
        self.weight = tensors["weight"]
        if (
            self.weight.shape != (128, 128)
            or self.weight.dtype != np.float32
            or not np.isfinite(self.weight).all()
        ):
            raise ValueError("Invalid adapter weight")
        self.identity = hashlib.sha256(
            (path / "adapter.json").read_bytes() + data
        ).hexdigest()

    def __call__(self, embeddings):
        import mlx.core as mx

        x = embeddings.astype(mx.float32) @ mx.array(self.weight).T
        return x / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), 1e-12)


def check_index_adapter(manifest, adapter):
    identity = adapter.identity if adapter else None
    if manifest.get("adapter_sha256") != identity:
        raise ValueError(
            "Query and index adapters differ; rebuild or use the same adapter"
        )
