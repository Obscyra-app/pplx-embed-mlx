"""Six inherited text layers, frozen backbone. Original weights stay untouched."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
from safetensors import safe_open
from ..model import Encoder

LAYERS = [0, 1, 6, 7, 10, 11]
BASE = Path("models/pplx-late-0.6b/source")
DEST = Path("models/pplx-late-0.6b/student6-base")


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


class Student(Encoder):
    def __init__(self, path=DEST):
        self.path = Path(path)
        m = json.loads((self.path / "student.json").read_text())
        if m["format"] != "pplx-student6-v1" or m["layers"] != LAYERS:
            raise ValueError("Unsupported student")
        if set(m["files"]) != {
            "student.safetensors",
            "config.json",
            "tokenizer.json",
            "2_MultiVectorMask/config.json",
        }:
            raise ValueError("Incomplete student file manifest")
        for name, h in m["files"].items():
            if digest(self.path / name) != h:
                raise ValueError("Student file checksum mismatch")
        self.config = json.loads((self.path / "config.json").read_text())
        assert self.config["text_config"]["num_hidden_layers"] == 6
        assert (
            self.config["text_config"]["layer_types"]
            == ["linear_attention", "full_attention"] * 3
        )
        self.weights = mx.load(str(self.path / "student.safetensors"))
        mx.eval(self.weights)
        self.dtype = mx.float16
        self.bits = None
        self.quant = {}
        self.used = set()

    def __call__(self, *args, **kwargs):
        if kwargs.get("pixel_values") is not None:
            raise ValueError("Text-only student")
        return super().__call__(*args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEST)
    ap.add_argument("--adapter", type=Path)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    weights = {}
    for file, prefix in [
        ("model.safetensors", ""),
        ("1_Dense/model.safetensors", "projection."),
    ]:
        with safe_open(str(BASE / file), framework="numpy") as f:
            for key in f.keys():
                k = prefix + key
                if k.startswith("visual."):
                    continue
                if k.startswith("language_model.layers."):
                    parts = k.split(".")
                    old = int(parts[2])
                    if old not in LAYERS:
                        continue
                    parts[2] = str(LAYERS.index(old))
                    k = ".".join(parts)
                dtype = (
                    mx.float32
                    if any(x in k for x in ["norm", "A_log", "dt_bias", "projection."])
                    else mx.float16
                )
                weights[k] = mx.array(f.get_tensor(key)).astype(dtype)
                mx.eval(weights[k])
                mx.clear_cache()
    if a.adapter:
        w = mx.array(np.load(a.adapter, allow_pickle=False))
        weights["projection.linear.weight"] = w @ weights["projection.linear.weight"]
        # Source projection has no bias; refuse silently dropping one.
        assert "projection.linear.bias" not in weights
    mx.save_safetensors(str(a.output / "student.safetensors"), weights)
    config = json.loads((BASE / "config.json").read_text())
    config["text_config"]["num_hidden_layers"] = 6
    config["text_config"]["layer_types"] = ["linear_attention", "full_attention"] * 3
    (a.output / "config.json").write_text(json.dumps(config, indent=2))
    for name in ["tokenizer.json", "2_MultiVectorMask/config.json"]:
        dest = a.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / name, dest)
    meta = {
        "format": "pplx-student6-v1",
        "layers": LAYERS,
        "text_only": True,
        "source_revision": "8fc2de24534aa3610d85fa59c463313a5f096455",
        "parameters": sum(v.size for v in weights.values()),
        "tensor_bytes": sum(v.nbytes for v in weights.values()),
        "trained_output_adapter_sha256": digest(a.adapter) if a.adapter else None,
        "files": {
            n: digest(a.output / n)
            for n in [
                "student.safetensors",
                "config.json",
                "tokenizer.json",
                "2_MultiVectorMask/config.json",
            ]
        },
    }
    (a.output / "student.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
