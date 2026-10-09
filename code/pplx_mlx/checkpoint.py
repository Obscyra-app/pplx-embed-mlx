"""Versioned MLX checkpoint export, integrity verification and schema validation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

FORMAT = "pplx-mlx-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load(path, dtype, bits):
    import mlx.core as mx

    root = Path(path)
    meta = json.loads((root / "mlx_checkpoint.json").read_text())
    if (
        meta["format"] != FORMAT
        or meta["dtype"] != dtype
        or meta["bits"] != bits
        or meta["group_size"] != 64
    ):
        raise ValueError(
            "Checkpoint precision/format mismatch; specify matching dtype/bits"
        )
    required = {
        "mlx_model.safetensors",
        "config.json",
        "tokenizer.json",
        "2_MultiVectorMask/config.json",
    }
    if set(meta["files"]) != required:
        raise ValueError("Incomplete checkpoint manifest")
    for name, digest in meta["files"].items():
        p = root / name
        if sha256(p) != digest:
            raise ValueError("Checkpoint integrity failed: " + name)
    values = mx.load(str(root / "mlx_model.safetensors"))
    mx.eval(values)
    actual = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in values.items()
    }
    if actual != meta["tensor_schema"]:
        raise ValueError("Unexpected checkpoint tensor schema")
    weights = {k: v for k, v in values.items() if not k.startswith("__quant__.")}
    quant = {}
    for name in meta["quantized_keys"]:
        quant[name] = tuple(
            values[f"__quant__.{name}.{suffix}"]
            for suffix in ("weight", "scales", "biases")
        )
    if set(values) != set(weights) | {
        f"__quant__.{name}.{suffix}"
        for name in quant
        for suffix in ("weight", "scales", "biases")
    }:
        raise ValueError("Unconsumed checkpoint tensor")
    return weights, quant


def main():
    import mlx.core as mx
    from .model import Encoder
    from .cli import fingerprint

    p = argparse.ArgumentParser()
    p.add_argument("--source", default="models/pplx-late-0.6b/source")
    p.add_argument("--output", required=True)
    p.add_argument("--bits", type=int, choices=[4, 6, 8])
    a = p.parse_args()
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=False)
    model = Encoder(a.source, "float16", a.bits)
    arrays = dict(model.weights)
    for name, (q, s, b) in model.quant.items():
        for suffix, v in zip(("weight", "scales", "biases"), (q, s, b)):
            arrays[f"__quant__.{name}.{suffix}"] = v
    mx.save_safetensors(str(dest / "mlx_model.safetensors"), arrays)
    names = ["config.json", "tokenizer.json", "2_MultiVectorMask/config.json"]
    for name in names:
        target = dest / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(a.source) / name, target)
    meta = {
        "format": FORMAT,
        "dtype": "float16",
        "bits": a.bits,
        "group_size": 64,
        "quantized_keys": sorted(model.quant),
        "tensor_schema": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in arrays.items()
        },
        "source_fingerprint": fingerprint(a.source),
        "files": {
            name: sha256(dest / name) for name in ["mlx_model.safetensors"] + names
        },
    }
    (dest / "mlx_checkpoint.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "output": str(dest),
                "bits": a.bits,
                "weight_bytes": (dest / "mlx_model.safetensors").stat().st_size,
                "sha256": meta["files"]["mlx_model.safetensors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
