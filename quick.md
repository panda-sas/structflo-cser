uv run python struct_labels/inference/detector.py --conf 0.60 --no_tile --grayscale
 --image_dir data/test_images/ --pair --max_dist 500


uv run python struct_labels/generation/dataset.py --smiles data/smiles/chembl_smiles.csv --out data/generate
d --num-train 3000 --num-val 500  --fonts-dir data/fonts/ --distractors-dir data/distractors/