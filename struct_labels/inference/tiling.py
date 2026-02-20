"""Sliding-window tiling for large-image inference."""


def generate_tiles(
    img_w: int,
    img_h: int,
    tile_size: int = 1536,
    overlap: float = 0.20,
) -> list[tuple]:
    """Generate tile coordinates (x1, y1, x2, y2) covering the full image."""
    step = int(tile_size * (1 - overlap))
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            tiles.append((x_start, y_start, x_end, y_end))
            if x_end >= img_w:
                break
            x += step
        if y_end >= img_h:
            break
        y += step
    return tiles
