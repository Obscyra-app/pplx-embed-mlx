"""Torch-free single-example preprocessing with bounded bicubic image resize."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from .resize import smart_size, bicubic_uint8


class Processor:
    def __init__(self, path):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.skip = {
            self.tokenizer.token_to_id(x)
            for x in json.loads((path / "2_MultiVectorMask/config.json").read_text())[
                "skiplist_words"
            ]
        }
        self.skip.discard(None)

    def __call__(self, value, task="document"):
        if task not in ("query", "document"):
            raise ValueError("task must be query or document")
        prefix = "[Q] " if task == "query" else "[D] "
        extra = {}
        if isinstance(value, Image.Image):
            if task != "document":
                raise ValueError("Image queries not validated")
            w, h = value.size
            nh, nw = smart_size(h, w)
            if (nh, nw) != (h, w):
                value = Image.fromarray(bicubic_uint8(value.convert("RGB"), nh, nw))
                h, w = nh, nw
            gh, gw = h // 16, w // 16
            n = gh * gw // 4
            text = prefix + "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>"
            # Match torchvision rescale_and_normalize and Qwen2VL patch order.
            a = (
                np.asarray(value.convert("RGB")).astype(np.float32).transpose(2, 0, 1)
                / 255
            )
            a = (a - 0.5) / 0.5
            a = a.reshape(3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(
                1, 4, 2, 5, 0, 3, 6
            )
            a = np.repeat(a[..., None, :, :], 2, axis=-3).reshape(gh * gw, -1)
            extra = {
                "pixel_values": a,
                "image_grid_thw": np.array([[1, gh, gw]], dtype=np.int64),
            }
        elif isinstance(value, str):
            text = prefix + value
        else:
            raise TypeError("Expected str or PIL.Image")
        ids = np.array(
            [self.tokenizer.encode(text, add_special_tokens=False).ids], dtype=np.int64
        )
        limit = 1024 if task == "query" else 4096
        if ids.shape[1] > limit:
            raise ValueError(
                f"Input exceeds {limit} tokens; explicit chunking required"
            )
        mask = np.ones_like(ids)
        pos = np.broadcast_to(np.arange(ids.shape[1]), (3, 1, ids.shape[1])).copy()
        if extra:
            start = int(np.flatnonzero(ids[0] == 248056)[0])
            mh, mw = gh // 2, gw // 2
            pos[:, 0, start : start + n] = (
                np.stack(
                    [
                        np.zeros(n, dtype=int),
                        np.repeat(np.arange(mh), mw),
                        np.tile(np.arange(mw), mh),
                    ]
                )
                + start
            )
            pos[:, 0, start + n :] = (
                np.arange(ids.shape[1] - start - n) + start + max(mh, mw)
            )
        outmask = mask.astype(bool)
        if task == "document":
            outmask &= ~np.isin(ids, list(self.skip))
        return dict(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            output_mask=outmask,
            **extra,
        )
