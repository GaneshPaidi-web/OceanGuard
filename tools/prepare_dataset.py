"""
tools/prepare_dataset.py
========================
Convert YOLO / VOC / COCO annotations into the COCO layout expected by
D-FINE, then generate a ready-to-run fine-tuning config.

Supported inputs
----------------
* ``yolo``  — images/{train,val} + labels/{train,val} + data.yaml
* ``voc``   — Pascal VOC XML files next to images
* ``coco``  — a COCO JSON (or train/val pair)

Output layout
-------------
    datasets/drowning/
        images/train/*.jpg
        images/val/*.jpg
        annotations/instances_train.json
        annotations/instances_val.json
        dataset_meta.json

It also writes:
    D-FINE/configs/dataset/drowning_detection.yml
    D-FINE/configs/dfine/custom/dfine_hgnetv2_s_drowning.yml

Class handling
--------------
``--class-mode person`` (default) merges every box into a single ``person``
class.  That is what the runtime pipeline needs: D-FINE only has to find
people in water, the risk engine decides what they are doing.

``--class-mode keep`` preserves source classes (e.g. Swimming / Struggling /
Drowning).  Category ids are written 0-based because D-FINE indexes its
classification logits directly with ``category_id``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DFINE_DIR = PROJECT_ROOT / "D-FINE"
DATASET_DIR = PROJECT_ROOT / "datasets"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def log(message: str):
    print(f"[prepare] {message}")


def find_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_data_yaml(path: Path) -> Dict:
    """Minimal YAML reader (avoids a hard PyYAML dependency for names)."""
    try:
        import yaml
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        # Fallback: parse only the names block.
        names = []
        text = path.read_text(encoding="utf-8", errors="ignore")
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                rest = stripped[len("names:"):].strip()
                if rest.startswith("["):
                    return {"names": [x.strip(" '\"") for x in rest.strip("[]").split(",")]}
                continue
            if in_names:
                if ":" in stripped:
                    value = stripped.split(":", 1)[1].strip().strip(" '\"")
                    if value:
                        names.append(value)
                elif stripped.startswith("-"):
                    names.append(stripped[1:].strip().strip(" '\""))
                elif stripped and not stripped.startswith("#"):
                    break
        return {"names": names}


def find_label_for_image(image: Path) -> Optional[Path]:
    parts = list(image.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        new_parts = parts[:idx] + ["labels"] + parts[idx + 1:]
        candidate = Path(*new_parts).with_suffix(".txt")
        if candidate.exists():
            return candidate

    sibling = image.with_suffix(".txt")
    if sibling.exists():
        return sibling

    return None


def coco_annotation(image_id: int, category_id: int, bbox: Sequence[float],
                    width: int, height: int) -> dict:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return {
        "id": 0,  # filled later
        "image_id": image_id,
        "category_id": int(category_id),
        "bbox": [x1, y1, w, h],
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [],
    }


def parse_yolo_label(label_path: Path, width: int, height: int) -> List[Tuple[int, List[float]]]:
    boxes = []
    try:
        text = label_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return boxes

    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            class_id = int(float(fields[0]))
            cx, cy, bw, bh = [float(v) for v in fields[1:5]]
        except ValueError:
            continue
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append((class_id, [x1, y1, x2, y2]))
    return boxes


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[str, List[float]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "person")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append((
            name,
            [
                float(bnd.findtext("xmin", 0)),
                float(bnd.findtext("ymin", 0)),
                float(bnd.findtext("xmax", 0)),
                float(bnd.findtext("ymax", 0)),
            ],
        ))
    return width, height, boxes


# --------------------------------------------------
# WRITERS
# --------------------------------------------------

def write_coco(
    entries: List[dict],
    categories: List[dict],
    destination: Path,
):
    payload = {
        "info": {"description": "Drowning detection dataset", "version": "1.0"},
        "licenses": [],
        "images": [e["image"] for e in entries],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 1
    for entry in entries:
        for ann in entry["annotations"]:
            ann["id"] = annotation_id
            annotation_id += 1
            payload["annotations"].append(ann)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    log(f"Wrote {destination} "
        f"({len(payload['images'])} images, {len(payload['annotations'])} boxes)")


def copy_image(image: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / image.name
    if target.exists() and target.stat().st_size == image.stat().st_size:
        return target
    shutil.copy2(image, target)
    return target


def build_entries(
    splits: Dict[str, List[Path]],
    output: Path,
    class_mode: str,
    class_names: List[str],
    category_map: Optional[Dict[str, int]] = None,
    source_format: str = "yolo",
):
    categories = (
        [{"id": 0, "name": "person", "supercategory": "person"}]
        if class_mode == "person"
        else [{"id": i, "name": name, "supercategory": "person"}
              for i, name in enumerate(class_names)]
    )

    for split, images in splits.items():
        entries = []
        for image_id, image_path in enumerate(images, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]

            boxes: List[Tuple[int, List[float]]] = []

            if source_format == "yolo":
                label_path = find_label_for_image(image_path)
                if label_path:
                    boxes = parse_yolo_label(label_path, width, height)
            elif source_format == "voc":
                xml_path = image_path.with_suffix(".xml")
                if xml_path.exists():
                    w_xml, h_xml, voc_boxes = parse_voc_xml(xml_path)
                    width = w_xml or width
                    height = h_xml or height
                    name_to_id = {name.lower(): i for i, name in enumerate(class_names)}
                    for name, box in voc_boxes:
                        boxes.append((name_to_id.get(name.lower(), 0), box))
            elif source_format == "coco":
                # Handled by the caller (already converted).
                continue

            target_image_dir = output / "images" / split
            target = copy_image(image_path, target_image_dir)

            annotations = []
            for source_class, box in boxes:
                if class_mode == "person":
                    category_id = 0
                else:
                    if category_map is not None:
                        category_id = category_map.get(str(source_class), source_class)
                    else:
                        category_id = int(source_class)
                    if category_id >= len(categories):
                        continue

                ann = coco_annotation(
                    image_id, category_id, box, width, height
                )
                if ann["area"] > 1.0:
                    annotations.append(ann)

            entries.append({
                "image": {
                    "id": image_id,
                    "file_name": target.name,
                    "width": width,
                    "height": height,
                },
                "annotations": annotations,
            })

        write_coco(entries, categories, output / "annotations" / f"instances_{split}.json")


# --------------------------------------------------
# CONFIG GENERATION
# --------------------------------------------------

def write_configs(output: Path, num_classes: int, model: str = "s"):
    dataset_yml = DFINE_DIR / "configs" / "dataset" / "drowning_detection.yml"
    dataset_yml.parent.mkdir(parents=True, exist_ok=True)

    images_train = (output / "images" / "train").resolve().as_posix()
    images_val = (output / "images" / "val").resolve().as_posix()
    ann_train = (output / "annotations" / "instances_train.json").resolve().as_posix()
    ann_val = (output / "annotations" / "instances_val.json").resolve().as_posix()

    dataset_yml.write_text(
        f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: {num_classes}
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {images_train}
    ann_file: {ann_train}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction


val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {images_val}
    ann_file: {ann_val}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
""",
        encoding="utf-8",
    )

    config_yml = DFINE_DIR / "configs" / "dfine" / "custom" / f"dfine_hgnetv2_{model}_drowning.yml"
    config_yml.parent.mkdir(parents=True, exist_ok=True)
    config_yml.write_text(
        f"""__include__: [
  '../../dataset/drowning_detection.yml',
  '../../runtime.yml',
  '../include/dataloader.yml',
  '../include/optimizer.yml',
  '../include/dfine_hgnetv2.yml',
]

output_dir: ./output/dfine_hgnetv2_{model}_drowning

DFINE:
  backbone: HGNetv2

HGNetv2:
  name: 'B0'
  return_idx: [1, 2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True

DFINETransformer:
  num_layers: 3
  eval_idx: -1

HybridEncoder:
  in_channels: [256, 512, 1024]
  hidden_dim: 256
  depth_mult: 0.34
  expansion: 0.5

epochs: 60
train_dataloader:
  total_batch_size: 16
  dataset:
    transforms:
      policy:
        epoch: 55
  collate_fn:
    stop_epoch: 55
    ema_restart_decay: 0.9999
    base_size_repeat: 20

val_dataloader:
  total_batch_size: 32
""",
        encoding="utf-8",
    )

    log(f"Wrote {dataset_yml}")
    log(f"Wrote {config_yml}")
    log("Fine-tune with:")
    log(f"  python tools/train_dfine.py --config {config_yml.as_posix()}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO/VOC/COCO data to D-FINE COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--format", choices=["yolo", "voc", "coco"], default="yolo")
    parser.add_argument("--source", required=True, help="Dataset root folder")
    parser.add_argument("--out", default=str(DATASET_DIR / "drowning"))
    parser.add_argument("--data-yaml", help="YOLO data.yaml (optional)")
    parser.add_argument("--class-mode", choices=["person", "keep"], default="person")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Used only when no val split exists")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--limit", type=int, default=0, help="Debug: cap images")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    class_names = ["person"]
    splits: Dict[str, List[Path]] = {}

    if args.format == "yolo":
        if args.data_yaml:
            data = read_data_yaml(Path(args.data_yaml))
            class_names = data.get("names", class_names)
            if isinstance(class_names, dict):
                class_names = [class_names[k] for k in sorted(class_names)]

        train_dir = source / "images" / "train"
        val_dir = source / "images" / "val"

        if train_dir.exists():
            splits["train"] = find_images(train_dir)
        if val_dir.exists():
            splits["val"] = find_images(val_dir)

        if not splits:
            # Single folder: split it.
            images = find_images(source)
            random.Random(args.seed).shuffle(images)
            cut = int(len(images) * (1.0 - args.val_split))
            splits = {"train": images[:cut], "val": images[cut:]}

        if not class_names or class_names == ["person"]:
            for candidate in ("classes.txt",):
                path = source / candidate
                if path.exists():
                    class_names = [line.strip() for line in
                                   path.read_text(encoding="utf-8",
                                                  errors="ignore").splitlines()
                                   if line.strip()]
                    break

    elif args.format == "voc":
        images = find_images(source)
        random.Random(args.seed).shuffle(images)
        cut = int(len(images) * (1.0 - args.val_split))
        splits = {"train": images[:cut], "val": images[cut:]}
        # Collect class names from the XML files.
        names = set()
        for image in images[:200]:
            xml_path = image.with_suffix(".xml")
            if xml_path.exists():
                try:
                    _, _, boxes = parse_voc_xml(xml_path)
                    names.update(name for name, _ in boxes)
                except Exception:
                    pass
        class_names = sorted(names) or ["person"]

    elif args.format == "coco":
        # Already COCO: copy and split if needed.
        images = find_images(source)
        random.Random(args.seed).shuffle(images)
        cut = int(len(images) * (1.0 - args.val_split))
        splits = {"train": images[:cut], "val": images[cut:]}
        log("COCO input detected. Re-using images; annotations are rebuilt "
            "from any sibling YOLO labels if present.")
        args.format = "yolo"

    if args.limit > 0:
        for split in splits:
            splits[split] = splits[split][:args.limit]

    for split, images in splits.items():
        log(f"{split}: {len(images)} images")

    if not splits.get("train"):
        raise SystemExit("No training images found.")

    if not splits.get("val"):
        splits["val"] = splits["train"][:max(1, len(splits["train"]) // 10)]

    build_entries(
        splits, output, args.class_mode, class_names,
        source_format=args.format,
    )

    num_classes = 1 if args.class_mode == "person" else len(class_names)
    write_configs(output, num_classes, model=args.model)

    (output / "dataset_meta.json").write_text(
        json.dumps({
            "classes": ["person"] if args.class_mode == "person" else class_names,
            "num_classes": num_classes,
            "splits": {k: len(v) for k, v in splits.items()},
            "source_format": args.format,
            "class_mode": args.class_mode,
        }, indent=2),
        encoding="utf-8",
    )

    log("Done.")
    log("Next steps:")
    log("  1. (Optional) download pretrained weights:")
    log("     https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth")
    log("  2. Fine-tune:")
    log("     python tools/train_dfine.py --config "
        f"{(DFINE_DIR / 'configs' / 'dfine' / 'custom' / f'dfine_hgnetv2_{args.model}_drowning.yml').as_posix()}")


if __name__ == "__main__":
    main()
