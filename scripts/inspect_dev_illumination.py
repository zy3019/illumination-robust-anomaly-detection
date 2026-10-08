"""Inspect clipping on development images only; keep every configured condition."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anomaly.data import file_hash, load_image, load_mask, perturb, save_json


def main():
    torch.set_num_threads(2)
    matrix = json.loads((ROOT / "configs/dev_matrix_v1.json").read_text())
    base = json.loads((ROOT / matrix["base_config"]).read_text())
    manifest = json.loads((ROOT / base["manifest"]).read_text())
    root = ROOT / base["data_root"]
    folder = ROOT / "reports/dev_matrix_v1"
    folder.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    overview = []
    displayed = ["clean", "brightness_0.5", "brightness_1.5", "contrast_0.5", "contrast_1.5"]
    for category in matrix["categories"]:
        rows = [r for r in manifest["rows"] if r["category"] == category and r["split"] == "dev"]
        representatives = {}
        for row in rows:
            image = load_image(root, row, base["image_size"])
            mask = load_mask(root, row, base["image_size"]).astype(bool)
            is_representative = row["defect"] not in representatives
            if is_representative:
                representatives[row["defect"]] = {"path": row["path"], "images": {}, "mask": mask}
            for condition in matrix["conditions"]:
                if condition["operation"] == "brightness":
                    unbounded = image * condition["factor"]
                elif condition["operation"] == "contrast":
                    mean = image.mean((-2, -1), keepdim=True)
                    unbounded = (image - mean) * condition["factor"] + mean
                else:
                    unbounded = image
                clipped = ((unbounded < 0) | (unbounded > 1)).any(dim=0).numpy()
                fraction = float(clipped.mean())
                defect_fraction = float(clipped[mask].mean()) if mask.any() else None
                diagnostics.append({"category": category, "path": row["path"], "defect": row["defect"],
                                    "label": row["label"], "condition": condition["name"],
                                    "fraction_pixels_with_any_channel_clipped": fraction,
                                    "fraction_defect_pixels_clipped": defect_fraction})
                if is_representative and condition["name"] in displayed:
                    array = perturb(image, condition["operation"], condition["factor"]).permute(1, 2, 0).numpy()
                    representatives[row["defect"]]["images"][condition["name"]] = Image.fromarray(np.uint8(array * 255))
        width, height = 170, 192
        montage = Image.new("RGB", (width * 6, 28 + height * len(representatives)), "white")
        draw = ImageDraw.Draw(montage)
        for column, title in enumerate([*displayed, "ground_truth"]):
            draw.text((column * width + 5, 5), title, fill="black")
        for index, (defect, info) in enumerate(representatives.items()):
            y = 28 + index * height
            for column, name in enumerate(displayed):
                montage.paste(info["images"][name].resize((160, 160)), (column * width + 5, y + 24))
            mask_img = Image.fromarray(np.uint8(info["mask"] * 255)).convert("RGB")
            montage.paste(mask_img.resize((160, 160), Image.Resampling.NEAREST), (5 * width + 5, y + 24))
            draw.text((5, y + 4), f'{defect}/{Path(info["path"]).name}', fill="black")
        montage.save(folder / f"{category}_illumination.png")
        overview.append({"category": category, "representatives": [r["path"] for r in representatives.values()]})
    with (folder / "clipping.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    save_json(folder / "illumination_inspection.json", {
        "manifest_sha256": file_hash(ROOT / base["manifest"]), "split": "dev",
        "definition": "fraction of pixels with any RGB channel outside [0,1] before clamp; equality at 0/1 is not clipping",
        "representative_rule": "first lexicographic dev path per defect type, including good",
        "note": "Clipping does not by itself establish loss of defect visibility. No condition removed.",
        "representatives": overview, "diagnostic_rows": len(diagnostics),
    })
    print(f"Saved {len(diagnostics)} clipping measurements and 3 contact sheets: {folder}")


if __name__ == "__main__":
    main()
