"""
tools/evaluate.py
=================
Offline quality checks for the two halves of the system.

Detector (D-FINE)
-----------------
Computes precision / recall / F1 / AP@0.5 for the person detector on a COCO
JSON or YOLO-format dataset::

    python tools/evaluate.py detector \
        --images datasets/drowning/images/val \
        --annotations datasets/drowning/annotations/instances_val.json \
        --onnx models/dfine_s_coco.onnx

    python tools/evaluate.py detector \
        --images datasets/drowning/images/val \
        --labels datasets/drowning/labels/val \
        --onnx models/dfine_s_drowning.onnx

Risk engine
-----------
Uses the feature dump produced by ``video_processor.py --dump-features``
after you fill its ``label`` column.  Reports per-track confusion and F1,
plus frame-level metrics::

    python tools/evaluate.py risk --features outputs/features/
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RISK_PRIORITY = {"UNKNOWN": 0, "NORMAL": 1, "DISTRESS": 2, "DROWNING": 3}


# --------------------------------------------------
# GEOMETRY
# --------------------------------------------------

def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# DATASETS
# --------------------------------------------------

def load_coco(path: Path) -> Dict[str, List[Tuple[int, List[float]]]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    images = {img["id"]: img["file_name"] for img in data.get("images", [])}
    result: Dict[str, List[Tuple[int, List[float]]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in images:
            continue
        x, y, w, h = ann["bbox"]
        result[images[image_id]].append(
            (int(ann.get("category_id", 0)), [x, y, x + w, y + h])
        )
    return result


def load_yolo(labels_dir: Path, image_dirs: Sequence[Path]) -> Dict[str, List[Tuple[int, List[float]]]]:
    import cv2

    # Map stem -> image path so we can recover image sizes.
    image_paths: Dict[str, Path] = {}
    for directory in image_dirs:
        for path in directory.rglob("*"):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                image_paths[path.stem] = path

    result: Dict[str, List[Tuple[int, List[float]]]] = defaultdict(list)
    for label_path in labels_dir.rglob("*.txt"):
        image_path = image_paths.get(label_path.stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            result[image_path.name].append((cls, [x1, y1, x2, y2]))
    return result


# --------------------------------------------------
# DETECTOR
# --------------------------------------------------

def _expand(paths: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.glob("*.csv")))
        elif any(ch in raw for ch in "*?[]"):
            found.extend(sorted(Path(p) for p in glob.glob(raw)))
        elif path.is_file():
            found.append(path)
    if not found:
        raise SystemExit("No matching files found.")
    return found


def evaluate_detector(args) -> int:
    import cv2

    from detection_pipeline import DetectionPipeline, PipelineConfig

    images_dir = Path(args.images)
    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found under {images_dir}")

    if args.annotations:
        ground_truth = load_coco(Path(args.annotations))
    elif args.labels:
        ground_truth = load_yolo(Path(args.labels), [images_dir])
    else:
        raise SystemExit("Pass --annotations (COCO JSON) or --labels (YOLO dir)")

    keep_classes = set(args.class_ids) if args.class_ids else None

    def filter_gt(entries):
        if keep_classes is None:
            return entries
        return [item for item in entries if item[0] in keep_classes]

    pipeline = DetectionPipeline(
        config=PipelineConfig(
            onnx_path=args.onnx,
            pose_model_path=args.pose,
            tta=args.tta,
            batch_size=args.batch_size,
            enable_sonar=False,
            enable_water_mask=(not args.no_water_mask),
            force_underwater=args.force_underwater,
        )
    )

    per_image: List[Tuple[List[Tuple[float, List[float]]], List[Tuple[int, List[float]]]]] = []
    total_gt = 0

    for index, path in enumerate(image_paths, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        pre = pipeline._preprocessor.process_detailed(  # noqa: SLF001
            frame, force_underwater=args.force_underwater
        )
        detections, _platform_boxes = pipeline._detect(  # noqa: SLF001
            pre.frame, pre.confidence_threshold, pre.scene_type
        )
        preds = [(float(d["confidence"]), [float(v) for v in d["box"]])
                 for d in detections]
        gt = filter_gt(ground_truth.get(path.name, []))
        total_gt += len(gt)
        per_image.append((preds, gt))
        if index % 10 == 0 or index == len(image_paths):
            print(f"\rDetector: {index}/{len(image_paths)} images", end="", file=sys.stderr)
    print("", file=sys.stderr)

    pipeline.close()

    # ---- confidence-ranked matching ----
    per_image_pred_count = [len(preds) for preds, _ in per_image]
    per_image_gt_count = [len(gt) for _, gt in per_image]

    # For AP we need a global ranking.  Use greedy matching at each IoU
    # threshold level independently.
    def match_at(iou_threshold: float) -> Tuple[List[float], List[int], List[int]]:
        scores: List[float] = []
        tps: List[int] = []
        gt_counts: List[int] = []
        for preds, gt in per_image:
            order = sorted(range(len(preds)), key=lambda i: preds[i][0], reverse=True)
            used = [False] * len(gt)
            for idx in order:
                score, box = preds[idx]
                best_iou, best_j = 0.0, -1
                for j, (_, gt_box) in enumerate(gt):
                    if used[j]:
                        continue
                    value = iou(box, gt_box)
                    if value > best_iou:
                        best_iou, best_j = value, j
                if best_j >= 0 and best_iou >= iou_threshold:
                    used[best_j] = True
                    scores.append(score)
                    tps.append(1)
                else:
                    scores.append(score)
                    tps.append(0)
            gt_counts.append(len(gt))
        return scores, tps, gt_counts

    scores, tps, gt_counts = match_at(0.5)
    if not scores:
        print("No detections produced.")
        return 1

    order = np.argsort(-np.asarray(scores))
    tps_sorted = np.asarray(tps)[order]
    tp_cum = np.cumsum(tps_sorted)
    fp_cum = np.cumsum(1 - tps_sorted)
    total_gt = int(sum(gt_counts))
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)

    # All-point (COCO-style) AP.
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum(np.diff(mrec) * mpre[:-1]))

    # Operating point at the requested confidence threshold.
    threshold = args.conf_threshold
    selected = [(score, tp) for score, tp in zip(scores, tps) if score >= threshold]
    tp_at = sum(tp for _, tp in selected)
    fp_at = len(selected) - tp_at
    fn_at = total_gt - tp_at
    precision = tp_at / max(1, tp_at + fp_at)
    recall = tp_at / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    print("\nDetector results")
    print(f"  images              : {len(image_paths)}")
    print(f"  ground-truth boxes  : {total_gt}")
    print(f"  detections          : {len(scores)}")
    print(f"  conf threshold      : {threshold:.2f}")
    print(f"  precision           : {precision:.3f}")
    print(f"  recall              : {recall:.3f}")
    print(f"  F1                  : {f1:.3f}")
    print(f"  AP@0.5              : {ap:.3f}")
    print(f"  TP/FP/FN            : {tp_at}/{fp_at}/{fn_at}")

    if args.export:
        payload = {
            "images": len(image_paths),
            "gt_boxes": total_gt,
            "detections": len(scores),
            "conf_threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap,
            "tp": tp_at,
            "fp": fp_at,
            "fn": fn_at,
            "per_image_gt": per_image_gt_count,
            "per_image_pred": per_image_pred_count,
        }
        Path(args.export).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {args.export}")

    return 0


# --------------------------------------------------
# RISK
# --------------------------------------------------

def _label(value: str) -> Optional[int]:
    text = (value or "").strip().lower()
    if text in ("1", "true", "yes", "drowning", "drown", "struggling"):
        return 1
    if text in ("0", "false", "no", "normal", "safe"):
        return 0
    return None


def evaluate_risk(args) -> int:
    files = _expand(args.features)

    rows: List[dict] = []
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = _label(row.get("label", ""))
                if label is None:
                    continue
                row["_label"] = label
                rows.append(row)

    if not rows:
        raise SystemExit(
            "No labelled rows found.  Fill the 'label' column of the feature "
            "CSV (1 = drowning/struggling, 0 = safe) first."
        )

    # -------- per track --------
    tracks: Dict[str, dict] = {}
    for row in rows:
        key = f"{row.get('source', '')}:{row.get('track_id', '0')}"
        state = tracks.setdefault(key, {
            "label": 0,
            "max_level": "UNKNOWN",
            "max_score": 0.0,
            "first_positive": None,
            "first_drowning": None,
            "rows": 0,
        })
        state["rows"] += 1
        state["label"] = max(state["label"], row["_label"])
        level = (row.get("risk_level") or "UNKNOWN").upper()
        if RISK_PRIORITY.get(level, 0) > RISK_PRIORITY.get(state["max_level"], 0):
            state["max_level"] = level
        try:
            state["max_score"] = max(state["max_score"], float(row.get("risk_score", 0)))
            timestamp = float(row.get("timestamp_seconds", 0))
        except ValueError:
            timestamp = 0.0
        if row["_label"] == 1 and state["first_positive"] is None:
            state["first_positive"] = timestamp
        if level == "DROWNING" and state["first_drowning"] is None:
            state["first_drowning"] = timestamp

    positives = [key for key, state in tracks.items() if state["label"] == 1]
    negatives = [key for key, state in tracks.items() if state["label"] == 0]
    decision_priority = RISK_PRIORITY[args.min_level]
    predicted = [
        key for key, state in tracks.items()
        if RISK_PRIORITY.get(state["max_level"], 0) >= decision_priority
    ]

    tp = len(set(positives) & set(predicted))
    fp = len(set(negatives) & set(predicted))
    fn = len(positives) - tp
    tn = len(negatives) - fp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    latencies = []
    for key in positives:
        state = tracks[key]
        if state["first_drowning"] is not None and state["first_positive"] is not None:
            latencies.append(state["first_drowning"] - state["first_positive"])

    # -------- frame level --------
    frame_tp = sum(1 for row in rows if row["_label"] == 1
                   and RISK_PRIORITY.get((row.get("risk_level") or "").upper(), 0)
                   >= RISK_PRIORITY.get("DROWNING", 0))
    frame_fn = sum(1 for row in rows if row["_label"] == 1) - frame_tp
    frame_fp = sum(1 for row in rows if row["_label"] == 0
                   and RISK_PRIORITY.get((row.get("risk_level") or "").upper(), 0)
                   >= RISK_PRIORITY.get("DROWNING", 0))
    frame_tn = sum(1 for row in rows if row["_label"] == 0) - frame_fp

    print("Risk results")
    print(f"  files                : {len(files)}")
    print(f"  labelled rows        : {len(rows)}")
    print(f"  tracks               : {len(tracks)} "
          f"({len(positives)} positive / {len(negatives)} negative)")
    print(f"  decision level       : >= {args.min_level}")
    print(f"  track precision      : {precision:.3f}")
    print(f"  track recall         : {recall:.3f}")
    print(f"  track F1             : {f1:.3f}")
    print(f"  track TP/FP/FN/TN    : {tp}/{fp}/{fn}/{tn}")
    print(f"  frame F1             : "
          f"{2 * frame_tp / max(1, 2 * frame_tp + frame_fp + frame_fn):.3f}")
    print(f"  frame TP/FP/FN/TN    : {frame_tp}/{frame_fp}/{frame_fn}/{frame_tn}")
    if latencies:
        print(f"  alert latency        : mean {np.mean(latencies):.2f}s, "
              f"median {np.median(latencies):.2f}s "
              f"(n={len(latencies)})")

    if args.verbose:
        print("\nPer track:")
        for key in sorted(tracks):
            state = tracks[key]
            print(f"  {key:<24} label={state['label']} "
                  f"max={state['max_level']:<8} score={state['max_score']:.2f} "
                  f"rows={state['rows']}")

    if args.export:
        payload = {
            "files": [str(f) for f in files],
            "rows": len(rows),
            "tracks": len(tracks),
            "positives": len(positives),
            "negatives": len(negatives),
            "min_level": args.min_level,
            "track": {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1,
            },
            "frame": {
                "tp": frame_tp, "fp": frame_fp,
                "fn": frame_fn, "tn": frame_tn,
            },
            "alert_latency": {
                "mean": float(np.mean(latencies)) if latencies else None,
                "median": float(np.median(latencies)) if latencies else None,
                "n": len(latencies),
            },
        }
        Path(args.export).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {args.export}")

    return 0


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the detector or the risk engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    det = sub.add_parser("detector", help="D-FINE precision/recall/AP")
    det.add_argument("--images", required=True)
    det.add_argument("--annotations", default=None, help="COCO JSON")
    det.add_argument("--labels", default=None, help="YOLO labels directory")
    det.add_argument("--onnx", default="models/dfine_s_coco.onnx")
    det.add_argument("--pose", default="models/pose_landmarker_lite.task")
    det.add_argument("--class-ids", type=int, nargs="*", default=[0],
                     help="Ground-truth classes to keep (default: person=0)")
    det.add_argument("--conf-threshold", type=float, default=0.35)
    det.add_argument("--batch-size", type=int, default=8)
    det.add_argument("--limit", type=int, default=0)
    det.add_argument("--tta", action="store_true")
    det.add_argument("--force-underwater", action="store_true")
    det.add_argument("--no-water-mask", action="store_true")
    det.add_argument("--export", default=None, help="Write metrics JSON here")
    det.set_defaults(func=evaluate_detector)

    risk = sub.add_parser("risk", help="risk engine track/frame metrics")
    risk.add_argument("--features", nargs="+", required=True,
                      help="Feature CSVs produced by --dump-features")
    risk.add_argument("--min-level", default="DROWNING",
                      choices=["DISTRESS", "DROWNING"])
    risk.add_argument("--verbose", action="store_true")
    risk.add_argument("--export", default=None)
    risk.set_defaults(func=evaluate_risk)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
