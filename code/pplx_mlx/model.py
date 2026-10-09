"""Inference-only Qwen3.5 bidirectional retriever. No Torch/Transformers imports."""

import json
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .gated_delta import gated_delta_update


def norm(x, w, eps=1e-6, offset=True):
    y = x.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + eps)
    return (y * (w.astype(mx.float32) + (1 if offset else 0))).astype(x.dtype)


def rope(x, c, s):
    d = c.shape[-1]
    a = x[..., :d]
    half = d // 2
    rot = mx.concatenate([-a[..., half:], a[..., :half]], axis=-1)
    return mx.concatenate([(a * c + rot * s).astype(x.dtype), x[..., d:]], axis=-1)


class Encoder:
    def __init__(self, path, dtype="float16", bits=None):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        if (
            self.config["model_type"] != "qwen3_5"
            or self.config["text_config"]["is_causal"]
        ):
            raise ValueError("Expected noncausal Qwen3.5")
        # Map one tensor at a time to avoid retaining a second full checkpoint.
        from safetensors import safe_open

        if dtype not in ("float32", "float16", "bfloat16") or bits not in (
            None,
            4,
            6,
            8,
        ):
            raise ValueError("Unsupported precision")
        tc = self.config["text_config"]
        vc = self.config["vision_config"]
        expected = {
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "head_dim": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_conv_kernel_dim": 4,
        }
        if (
            any(tc.get(k) != v for k, v in expected.items())
            or tc["layer_types"] != ["linear_attention", "full_attention"] * 6
            or vc["spatial_merge_size"] != 2
            or vc["num_position_embeddings"] != 2304
        ):
            raise ValueError(
                "Unsupported architecture; only the late-0.6B layout is implemented"
            )
        self.dtype = getattr(mx, dtype)
        self.weights = {}
        self.quant = {}
        self.bits = bits
        if (self.path / "mlx_checkpoint.json").exists():
            from .checkpoint import load

            self.weights, self.quant = load(self.path, dtype, bits)
            self.used = set()
            return
        for file, prefix in [
            ("model.safetensors", ""),
            ("1_Dense/model.safetensors", "projection."),
        ]:
            with safe_open(str(self.path / file), framework="numpy") as source:
                for key in source.keys():
                    k = prefix + key
                    target = (
                        mx.float32
                        if any(
                            s in k for s in ("A_log", "dt_bias", "norm", "projection.")
                        )
                        else self.dtype
                    )
                    value = mx.array(source.get_tensor(key)).astype(target)
                    mx.eval(value)
                    if (
                        bits
                        and k.startswith("language_model.layers.")
                        and value.ndim == 2
                        and ".mlp." in k
                    ):
                        self.quant[k] = mx.quantize(value, group_size=64, bits=bits)
                        mx.eval(self.quant[k])
                    else:
                        self.weights[k] = value
                    del value
                    mx.clear_cache()
        self.used = set()

    def w(self, k):
        self.used.add(k)
        return self.weights[k]

    def linear(self, x, p):
        k = p + ".weight"
        self.used.add(k)
        if k in self.quant:
            q, s, b = self.quant[k]
            y = mx.quantized_matmul(
                x, q, s, b, transpose=True, group_size=64, bits=self.bits
            )
        else:
            w = self.w(k)
            y = x.astype(w.dtype) @ w.T
        if p + ".bias" in self.weights:
            y = y + self.w(p + ".bias")
        return y

    def ln(self, x, p):
        return mx.fast.layer_norm(x, self.w(p + ".weight"), self.w(p + ".bias"), 1e-6)

    def rms(self, x, p):
        return norm(x, self.w(p + ".weight"))

    def vision(self, pixels, grids):
        vc = self.config["vision_config"]
        h = vc["num_heads"]
        d = vc["hidden_size"] // h
        w = self.w("visual.patch_embed.proj.weight").reshape(vc["hidden_size"], -1)
        x = pixels.astype(w.dtype) @ w.T + self.w("visual.patch_embed.proj.bias")
        pos = []
        positions = []
        lengths = []
        for t, gh, gw in np.asarray(grids).tolist():
            if t != 1:
                raise ValueError("Only still images supported")
            y, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")

            def order(z):
                return (
                    z.reshape(gh // 2, 2, gw // 2, 2).transpose(0, 2, 1, 3).reshape(-1)
                )

            yp, xp = order(y), order(xx)
            positions.append(np.stack([yp, xp], axis=-1))
            lengths.append(gh * gw)
            fy = yp.astype(np.float32) * (47 / max(gh - 1, 1))
            fx = xp.astype(np.float32) * (47 / max(gw - 1, 1))
            y0 = np.floor(fy).astype(int)
            x0 = np.floor(fx).astype(int)
            y1 = np.minimum(y0 + 1, 47)
            x1 = np.minimum(x0 + 1, 47)
            wy = mx.array(fy - y0, dtype=mx.float32)[:, None]
            wx = mx.array(fx - x0, dtype=mx.float32)[:, None]
            table = self.w("visual.pos_embed.weight")
            p = (
                table[mx.array(y0 * 48 + x0)] * (1 - wy) * (1 - wx)
                + table[mx.array(y0 * 48 + x1)] * (1 - wy) * wx
                + table[mx.array(y1 * 48 + x0)] * wy * (1 - wx)
                + table[mx.array(y1 * 48 + x1)] * wy * wx
            )
            pos.append(p)
        x = x + mx.concatenate(pos).astype(x.dtype)
        coords = mx.array(np.concatenate(positions), dtype=mx.float32)
        inv = 1 / (10000 ** (mx.arange(0, d // 2, 2, dtype=mx.float32) / (d // 2)))
        freq = (coords[..., None] * inv).reshape(-1, d // 2)
        freq = mx.concatenate([freq, freq], axis=-1)
        c, s = mx.cos(freq)[:, None, :], mx.sin(freq)[:, None, :]
        for i in range(vc["depth"]):
            p = f"visual.blocks.{i}"
            z = self.ln(x, p + ".norm1")
            qkv = self.linear(z, p + ".attn.qkv").reshape(-1, 3, h, d)
            q = rope(qkv[:, 0], c, s)
            k = rope(qkv[:, 1], c, s)
            v = qkv[:, 2]
            out = []
            offset = 0
            for n in lengths:
                a, b, e = [
                    z[offset : offset + n].transpose(1, 0, 2)[None] for z in (q, k, v)
                ]
                o = (
                    mx.fast.scaled_dot_product_attention(a, b, e, scale=d**-0.5)
                    .transpose(0, 2, 1, 3)
                    .reshape(n, -1)
                )
                out.append(o)
                offset += n
            x = x + self.linear(mx.concatenate(out), p + ".attn.proj")
            z = self.ln(x, p + ".norm2")
            z = nn.gelu_approx(self.linear(z, p + ".mlp.linear_fc1"))
            x = x + self.linear(z, p + ".mlp.linear_fc2")
            mx.eval(x)
        x = self.ln(x, "visual.merger.norm").reshape(-1, vc["hidden_size"] * 4)
        return self.linear(
            nn.gelu(self.linear(x, "visual.merger.linear_fc1")),
            "visual.merger.linear_fc2",
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        pixel_values=None,
        image_grid_thw=None,
        trace=False,
        **kwargs,
    ):
        if (
            np.asarray(input_ids).ndim != 2
            or np.asarray(input_ids).shape[0] != 1
            or not np.asarray(attention_mask).all()
        ):
            raise ValueError("Only one unpadded input per call is validated")
        ids = mx.array(input_ids)
        mask = mx.array(attention_mask).astype(mx.bool_)
        pos = mx.array(position_ids)
        tc = self.config["text_config"]
        b, seq_len = ids.shape
        hd = tc["head_dim"]
        nq = tc["num_attention_heads"]
        nkv = tc["num_key_value_heads"]
        x = self.w("language_model.embed_tokens.weight")[ids]
        if pixel_values is not None:
            visual = self.vision(mx.array(pixel_values), image_grid_thw)
            flat = x.reshape(-1, x.shape[-1])
            idx = np.flatnonzero(
                np.asarray(input_ids).reshape(-1) == self.config["image_token_id"]
            )
            if len(idx) != visual.shape[0]:
                raise ValueError("Image token/feature mismatch")
            flat[mx.array(idx)] = visual
            x = flat.reshape(x.shape)
        inv = 1 / (
            tc["rope_parameters"]["rope_theta"]
            ** (mx.arange(0, 64, 2, dtype=mx.float32) / 64)
        )
        if pos.shape[0] == 4:
            pos = pos[1:]
        freq = pos[..., None].astype(mx.float32) * inv
        freq0 = freq[0]
        for axis, offset in ((1, 1), (2, 2)):
            end = tc["rope_parameters"]["mrope_section"][axis] * 3
            freq0[..., offset:end:3] = freq[axis, ..., offset:end:3]
        freq0 = mx.concatenate([freq0, freq0], axis=-1)
        c = mx.cos(freq0)[:, None]
        s = mx.sin(freq0)[:, None]
        attn_mask = mx.broadcast_to(mask[:, None, None, :], (b, 1, seq_len, seq_len))
        states = []
        for i, kind in enumerate(tc["layer_types"]):
            p = f"language_model.layers.{i}"
            z = self.rms(x, p + ".input_layernorm")
            if kind == "full_attention":
                a = p + ".self_attn"
                q, g = mx.split(
                    self.linear(z, a + ".q_proj").reshape(b, seq_len, nq, hd * 2),
                    2,
                    axis=-1,
                )
                q = self.rms(q, a + ".q_norm").transpose(0, 2, 1, 3)
                k = self.rms(
                    self.linear(z, a + ".k_proj").reshape(b, seq_len, nkv, hd),
                    a + ".k_norm",
                ).transpose(0, 2, 1, 3)
                v = (
                    self.linear(z, a + ".v_proj")
                    .reshape(b, seq_len, nkv, hd)
                    .transpose(0, 2, 1, 3)
                )
                q, k = rope(q, c, s), rope(k, c, s)
                out = (
                    mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=hd**-0.5, mask=attn_mask
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape(b, seq_len, -1)
                )
                out = self.linear(
                    out * mx.sigmoid(g.reshape(b, seq_len, -1)), a + ".o_proj"
                )
            else:
                a = p + ".linear_attn"
                z = z * mask[..., None]
                qkv = self.linear(z, a + ".in_proj_qkv")
                cw = self.w(a + ".conv1d.weight")[:, 0, :].T
                padded = mx.pad(qkv, [(0, 0), (3, 0), (0, 0)])
                qkv = nn.silu(sum(padded[:, j : j + seq_len] * cw[j] for j in range(4)))
                q, k, v = [
                    a.reshape(b, seq_len, 16, 128) for a in mx.split(qkv, 3, axis=-1)
                ]
                q = (
                    q
                    * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                    * (128**-0.5)
                )
                k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                out, _ = gated_delta_update(
                    q,
                    k,
                    v,
                    self.linear(z, a + ".in_proj_a").astype(mx.float32),
                    self.linear(z, a + ".in_proj_b"),
                    self.w(a + ".A_log"),
                    self.w(a + ".dt_bias"),
                )
                gate = self.linear(z, a + ".in_proj_z").reshape(b, seq_len, 16, 128)
                out = norm(out, self.w(a + ".norm.weight"), offset=False) * nn.silu(
                    gate.astype(mx.float32)
                )
                out = self.linear(
                    out.reshape(b, seq_len, -1).astype(z.dtype), a + ".out_proj"
                )
            x = x + out
            z = self.rms(x, p + ".post_attention_layernorm")
            x = x + self.linear(
                nn.silu(self.linear(z, p + ".mlp.gate_proj"))
                * self.linear(z, p + ".mlp.up_proj"),
                p + ".mlp.down_proj",
            )
            mx.eval(x)
            if trace:
                states.append(np.asarray(x).copy())
        hidden = self.rms(x, "language_model.norm")
        out = self.linear(hidden, "projection.linear")
        out = out.astype(mx.float32)
        out = out / mx.maximum(
            mx.sqrt(mx.sum(out * out, axis=-1, keepdims=True)), 1e-12
        )
        mx.eval(out)
        return {"embeddings": out, "hidden": hidden, "states": states}
