"""Synthetic shape/noise patch generators used as distractor images."""

import random

import numpy as np
from PIL import Image, ImageDraw


def _gen_geometric_shapes(width: int, height: int) -> Image.Image:
    """Generate an image with random geometric shapes (simulates diagrams/logos)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_shapes = random.randint(3, 10)
    colors = [
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
        (0, 172, 193),
        (200, 200, 200),
        (100, 100, 100),
    ]
    for _ in range(n_shapes):
        shape = random.choice(["rect", "ellipse", "line", "triangle"])
        color = random.choice(colors)
        if shape == "rect":
            rx0 = random.randint(0, width - 20)
            ry0 = random.randint(0, height - 20)
            rx1 = rx0 + random.randint(10, min(80, width - rx0))
            ry1 = ry0 + random.randint(10, min(80, height - ry0))
            if random.random() < 0.5:
                draw.rectangle([rx0, ry0, rx1, ry1], fill=color)
            else:
                draw.rectangle([rx0, ry0, rx1, ry1], outline=color, width=2)
        elif shape == "ellipse":
            cx = random.randint(10, width - 10)
            cy = random.randint(10, height - 10)
            rx = random.randint(5, 40)
            ry = random.randint(5, 40)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
        elif shape == "line":
            lx0 = random.randint(0, width)
            ly0 = random.randint(0, height)
            lx1 = random.randint(0, width)
            ly1 = random.randint(0, height)
            draw.line([lx0, ly0, lx1, ly1], fill=color, width=random.randint(1, 3))
        elif shape == "triangle":
            pts = [
                (random.randint(0, width), random.randint(0, height)) for _ in range(3)
            ]
            draw.polygon(pts, fill=color)
    return img


def _gen_noise_patch(width: int, height: int) -> Image.Image:
    """Generate a patch of random noise (simulates a scan artifact or photo region)."""
    arr = np.random.randint(180, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _gen_gradient_block(width: int, height: int) -> Image.Image:
    """Generate a gradient block (simulates a shaded region or header bar)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    c1 = np.array([random.randint(180, 240)] * 3, dtype=np.float64)
    c2 = np.array([random.randint(220, 255)] * 3, dtype=np.float64)
    for y in range(height):
        t = y / max(1, height - 1)
        row_color = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        arr[y, :] = row_color
    return Image.fromarray(arr)
