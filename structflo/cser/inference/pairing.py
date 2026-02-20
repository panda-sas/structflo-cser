"""Hungarian matching to pair detected structures with their labels."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def centroid(bbox: list) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def pair_detections(
    detections: list[dict],
    max_distance: float | None = None,
) -> list[dict]:
    """Match each structure to its nearest label via Hungarian matching on centroid distance.

    Returns a list of pairs:
        [{"structure": det, "label": det, "distance": float}, ...]

    Unmatched detections (when counts differ) are silently dropped.
    If max_distance is set, pairs beyond that pixel distance are also dropped.
    """
    structures = [d for d in detections if d["class_id"] == 0]
    labels = [d for d in detections if d["class_id"] == 1]

    if not structures or not labels:
        return []

    cost = np.zeros((len(structures), len(labels)))
    for i, s in enumerate(structures):
        sc = centroid(s["bbox"])
        for j, lb in enumerate(labels):
            lc = centroid(lb["bbox"])
            cost[i, j] = np.hypot(sc[0] - lc[0], sc[1] - lc[1])

    row_ind, col_ind = linear_sum_assignment(cost)

    pairs = []
    for r, c in zip(row_ind, col_ind):
        dist = cost[r, c]
        if max_distance is None or dist <= max_distance:
            pairs.append(
                {
                    "structure": structures[r],
                    "label": labels[c],
                    "distance": dist,
                }
            )
    return pairs
