# Publishing New Model Weights

This guide is for developers who have retrained a model and need to make the new
weights available to users of the `structflo-cser` package.

---

## Concepts

Two things are versioned independently:

| Thing | Where | Example |
|---|---|---|
| Python package | PyPI | `structflo-cser==0.2.0` |
| Model weights | HuggingFace Hub | `weights-v1.1` (git tag) |

A weights release does **not** require a code/PyPI release unless the model
architecture changed (different classes, input shape, head structure).

---

## One-time setup: create the HF Hub repo

Do this once per model, not per release.

```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF write token

# Create the repo (public)
huggingface-cli repo create structflo/cser-detector --type model
```

---

## Case 1: Retrain on more data, same architecture

This is the common case — better accuracy, same YOLO11l 2-class model.

### Step 1 — upload the weights

```bash
# From the project root, after training completes:
python - <<'EOF'
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="runs/labels_detect/yolo11l_panels/weights/best.pt",
    path_in_repo="best.pt",
    repo_id="structflo/cser-detector",
    repo_type="model",
    commit_message="weights v1.1: retrained on 4000 pages",
)
EOF
```

### Step 2 — tag the commit on HF Hub

```bash
python - <<'EOF'
from huggingface_hub import HfApi

api = HfApi()
# Get the latest commit hash on main
info = api.repo_info(repo_id="structflo/cser-detector", repo_type="model")
commit_sha = info.sha

api.create_tag(
    repo_id="structflo/cser-detector",
    repo_type="model",
    tag="weights-v1.1",
    tag_message="weights v1.1: retrained on 4000 pages",
    revision=commit_sha,
)
EOF
```

### Step 3 — update `weights.py`

Open [structflo/cser/weights.py](../structflo/cser/weights.py) and add the new
entry to `REGISTRY["cser-detector"]`, then bump `LATEST`:

```python
REGISTRY: dict[str, dict[str, dict]] = {
    "cser-detector": {
        "v1.0": {
            "repo_id":  "structflo/cser-detector",
            "filename": "best.pt",
            "revision": "weights-v1.0",
            "sha256":   "abc123...",
            "requires": ">=0.1.0,<1.0.0",
        },
        # Add the new entry:
        "v1.1": {
            "repo_id":  "structflo/cser-detector",
            "filename": "best.pt",
            "revision": "weights-v1.1",
            "sha256":   "def456...",      # see note below
            "requires": ">=0.1.0,<1.0.0", # same architecture = same range
        },
    },
}

LATEST = {
    "cser-detector": "v1.1",   # <-- bump this
}
```

> **Getting the sha256:** After uploading, run:
> ```bash
> python -c "
> import hashlib, pathlib
> data = pathlib.Path('runs/labels_detect/yolo11l_panels/weights/best.pt').read_bytes()
> print(hashlib.sha256(data).hexdigest())
> "
> ```

### Step 4 — commit and push

```bash
git add structflo/cser/weights.py
git commit -m "weights: publish v1.1"
git push
```

No PyPI release is required. Users will get the new weights automatically on
their next run once they pull the updated package.

---

## Case 2: Architecture change (new classes, different model)

This requires both a new weights major version **and** a new package version,
because the old code cannot load the new weights.

### Step 1 — train and upload as above, but use `weights-v2.0`

Follow the same upload + tag steps, tagging as `weights-v2.0`.

### Step 2 — update `weights.py` with a **new `requires` range**

```python
"v2.0": {
    "repo_id":  "structflo/cser-detector",
    "filename": "best.pt",
    "revision": "weights-v2.0",
    "sha256":   "...",
    "requires": ">=1.0.0,<2.0.0",   # <-- requires the new package major
},
```

### Step 3 — bump the package version

In `pyproject.toml`:
```toml
version = "1.0.0"
```

### Step 4 — commit, tag, and publish to PyPI

```bash
git add structflo/cser/weights.py pyproject.toml
git commit -m "v1.0.0: 3-class detector"
git tag v1.0.0
git push && git push --tags
uv build && uv publish
```

Users on the old package who try to load `v2.0` weights will see:

```
WeightsCompatibilityError: Weights 'cser-detector/v2.0' require
structflo-cser>=1.0.0 (you have 0.2.0).
Upgrade with:  uv add 'structflo-cser>=1.0.0'
```

---

## Adding a new model entirely

For a second model (e.g. `ner-tagger`):

1. Create a new HF Hub repo: `structflo/ner-tagger`
2. Upload weights and tag as `weights-v1.0`
3. Add a top-level key to `REGISTRY` and `LATEST` in `weights.py`:

```python
REGISTRY = {
    "cser-detector": { ... },    # existing
    "ner-tagger": {
        "v1.0": {
            "repo_id":  "structflo/ner-tagger",
            "filename": "model.pt",
            "revision": "weights-v1.0",
            "requires": ">=0.2.0,<1.0.0",
        },
    },
}

LATEST = {
    "cser-detector": "v1.1",
    "ner-tagger":    "v1.0",    # <-- add
}
```

4. Call `resolve_weights("ner-tagger")` wherever that model is loaded.

---

## Decision tree

```
Did anything in the model architecture change?
(classes, imgsz, backbone, head)
│
├── No  → Case 1: new weights tag only, no PyPI release
│         bump LATEST, same "requires" range
│
└── Yes → Case 2: new weights major + new package major
          update "requires" to the new pkg version range
          publish to PyPI
```

---

## Checklist

- [ ] `best.pt` uploaded to HF Hub
- [ ] Commit tagged on HF Hub (`weights-vX.Y`)
- [ ] New entry added to `REGISTRY` in `weights.py`
- [ ] `LATEST["<model>"]` updated
- [ ] `sha256` recorded in registry entry
- [ ] `requires` specifier correct (same range if same arch, new range if arch changed)
- [ ] If arch changed: `pyproject.toml` version bumped and published to PyPI
