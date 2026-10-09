"""Torch CPU uint8 bicubic AA arithmetic (PyTorch BSD license in licenses/)."""

import math
import numpy as np


def smart_size(height, width, min_pixels=65536, max_pixels=262144):
    if min(height, width) < 1 or max(height, width) / min(height, width) > 200:
        raise ValueError("Invalid image dimensions/aspect ratio")
    h = round(height / 32) * 32
    w = round(width / 32) * 32
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(32, math.floor(height / beta / 32) * 32)
        w = max(32, math.floor(width / beta / 32) * 32)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / 32) * 32
        w = math.ceil(width * beta / 32) * 32
    return h, w


def bicubic_uint8(image, height, width):
    value = np.asarray(image, dtype=np.uint8)

    def cubic(x):
        x = np.abs(x)
        return np.where(
            x < 1,
            ((1.5 * x - 2.5) * x) * x + 1,
            np.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0),
        )

    for axis, target in ((1, width), (0, height)):
        original = value.shape[axis]
        if original == target:
            continue
        scale = original / target
        inv = 1 / max(scale, 1)
        support = 2 * max(scale, 1)
        kernels = []
        for i in range(target):
            center = (i + 0.5) * scale
            lo = max(int(center - support + 0.5), 0)
            hi = min(int(center + support + 0.5), original)
            indices = np.arange(lo, hi)
            weights = cubic((indices - center + 0.5) * inv)
            weights /= weights.sum()
            kernels.append((indices, weights))
        maximum = max(float(w.max()) for _, w in kernels)
        precision = 0
        while precision < 22 and int(0.5 + maximum * (1 << (precision + 1))) < 32768:
            precision += 1
        rows = []
        for indices, weights in kernels:
            coefficients = (
                weights * (1 << precision) + np.where(weights < 0, -0.5, 0.5)
            ).astype(np.int64)
            v = np.take(value, indices, axis=axis).astype(np.int64)
            shape = [1] * v.ndim
            shape[axis] = len(coefficients)
            total = (v * coefficients.reshape(shape)).sum(axis=axis) + (
                1 << (precision - 1)
            )
            rows.append(np.clip(total >> precision, 0, 255).astype(np.uint8))
        value = np.stack(rows, axis=axis)
    return value
