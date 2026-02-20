"""Synthetic chart generators used as distractor images on document pages."""

import random

from PIL import Image, ImageDraw


def _gen_bar_chart(width: int, height: int) -> Image.Image:
    """Generate a simple random bar chart image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_bars = random.randint(3, 8)
    bar_w = max(4, (width - 40) // n_bars - 4)
    max_bar_h = height - 40
    colors = [
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
        (0, 172, 193),
        (255, 112, 67),
        (117, 117, 117),
    ]
    x_start = 20
    for i in range(n_bars):
        bar_h = random.randint(max_bar_h // 5, max_bar_h)
        color = random.choice(colors)
        bx = x_start + i * (bar_w + 4)
        by = height - 20 - bar_h
        draw.rectangle([bx, by, bx + bar_w, height - 20], fill=color, outline=(0, 0, 0))
    # Axes
    draw.line((18, 15, 18, height - 18), fill=(0, 0, 0), width=2)
    draw.line((18, height - 18, width - 5, height - 18), fill=(0, 0, 0), width=2)
    return img


def _gen_scatter_plot(width: int, height: int) -> Image.Image:
    """Generate a simple random scatter plot image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_pts = random.randint(15, 60)
    color = random.choice(
        [
            (66, 133, 244),
            (219, 68, 55),
            (15, 157, 88),
            (171, 71, 188),
        ]
    )
    for _ in range(n_pts):
        cx = random.randint(25, width - 15)
        cy = random.randint(15, height - 25)
        r = random.randint(2, 5)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    # Axes
    draw.line((20, 10, 20, height - 18), fill=(0, 0, 0), width=2)
    draw.line((20, height - 18, width - 5, height - 18), fill=(0, 0, 0), width=2)
    return img


def _gen_line_plot(width: int, height: int) -> Image.Image:
    """Generate a simple random line plot image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_lines = random.randint(1, 3)
    colors = [(66, 133, 244), (219, 68, 55), (15, 157, 88)]
    for li in range(n_lines):
        n_pts = random.randint(5, 15)
        pts = []
        for i in range(n_pts):
            px = 25 + int(i * (width - 40) / max(1, n_pts - 1))
            py = random.randint(15, height - 25)
            pts.append((px, py))
        color = colors[li % len(colors)]
        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=color, width=2)
    draw.line((20, 10, 20, height - 18), fill=(0, 0, 0), width=2)
    draw.line((20, height - 18, width - 5, height - 18), fill=(0, 0, 0), width=2)
    return img


def _gen_pie_chart(size: int) -> Image.Image:
    """Generate a simple random pie chart image."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_slices = random.randint(3, 7)
    values = [random.random() for _ in range(n_slices)]
    total = sum(values)
    colors = [
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
        (0, 172, 193),
        (255, 112, 67),
    ]
    margin = 10
    bbox = [margin, margin, size - margin, size - margin]
    start = 0
    for i, v in enumerate(values):
        extent = (v / total) * 360
        draw.pieslice(
            bbox, start, start + extent, fill=colors[i % len(colors)], outline=(0, 0, 0)
        )
        start += extent
    return img
