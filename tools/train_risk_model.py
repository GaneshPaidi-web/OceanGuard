"""
tools/train_risk_model.py
=========================
Train the optional learned risk calibration used by
:class:`risk_engine.RiskEngine`.

The rule engine is already a solid baseline.  This script lets you replace
part of its hand-tuned weighting with a small gradient-boosted model trained
on labelled per-track features.

Data collection
---------------
Run the pipeline once per video and dump features::

    python video_processor.py test_images/beach.mp4 \
        --output-dir outputs --sample-fps 4 \
        --dump-features outputs/features/beach.csv

The CSV has one row per detection and an empty ``label`` column.  Fill it
with ``1`` (drowning / struggling) or ``0`` (safe) for the rows you care
about, then train::

    python tools/train_risk_model.py --features outputs/features/

If you do not have manual labels yet, ``--pseudo-labels`` derives weak labels
from the rule engine (DROWNING -> 1, NORMAL -> 0, everything else skipped).
That is *self-distillation*: it mostly smooths the existing rules and is a
useful sanity check, not a replacement for real annotations.

Output
------
``models/risk_model.pkl`` containing::

    {"model": HistGradientBoostingClassifier,
     "features": [...],
     "threshold": 0.5,
     "metrics": {...},
     "label_mode": "manual" | "pseudo",
     "trained_at": "..."}

``RiskEngine`` loads it automatically and blends ``0.6 * rules + 0.4 * model``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_engine import FEATURE_ORDER  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "risk_model.pkl"

POSITIVE_WORDS = {"1", "true", "yes", "y", "drowning", "drown", "struggling",
                  "distress", "risk", "positive"}
NEGATIVE_WORDS = {"0", "false", "no", "n", "normal", "safe", "negative"}


# --------------------------------------------------
# DATA
# --------------------------------------------------

def _expand_paths(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif any(ch in raw for ch in "*?[]"):
            files.extend(sorted(Path(p) for p in glob.glob(raw)))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"Feature file not found: {path}")
    unique: List[Path] = []
    seen = set()
    for path in files:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise SystemExit("No CSV feature files found.")
    return unique


def _parse_label(value: str) -> Optional[int]:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text in POSITIVE_WORDS:
        return 1
    if text in NEGATIVE_WORDS:
        return 0
    try:
        numeric = float(text)
    except ValueError:
        return None
    return 1 if numeric >= 0.5 else 0


def _pseudo_label(row: Dict[str, str]) -> Optional[int]:
    level = (row.get("risk_level") or "").strip().upper()
    if level == "DROWNING":
        return 1
    if level in ("NORMAL", "DISTRESS"):
        return 0
    return None


def load_rows(
    files: Sequence[Path],
    feature_columns: Sequence[str],
    pseudo_labels: bool,
    min_water_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    X: List[List[float]] = []
    y: List[int] = []
    groups: List[str] = []
    stats = Counter()

    for path in files:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [name for name in feature_columns if name not in reader.fieldnames]
            if missing:
                raise SystemExit(
                    f"{path} is missing feature columns: {missing}\n"
                    "Regenerate it with video_processor.py --dump-features."
                )
            for row in reader:
                stats["rows"] += 1
                label = _parse_label(row.get("label", ""))
                if label is None and pseudo_labels:
                    label = _pseudo_label(row)
                    stats["pseudo_labelled"] += 1
                if label is None:
                    stats["unlabelled"] += 1
                    continue

                if label == 0 and min_water_ratio > 0:
                    try:
                        water = float(row.get("water_ratio", "-1") or -1)
                    except ValueError:
                        water = -1.0
                    if water < min_water_ratio:
                        stats["land_background"] += 1
                        continue

                values = []
                valid = True
                for name in feature_columns:
                    try:
                        value = float(row.get(name, "nan"))
                    except ValueError:
                        value = math.nan
                    if not math.isfinite(value):
                        valid = False
                        break
                    values.append(value)
                if not valid:
                    stats["invalid_features"] += 1
                    continue

                source = row.get("source") or path.stem
                track = row.get("track_id") or "0"
                X.append(values)
                y.append(label)
                groups.append(f"{source}:{track}")
                stats["used"] += 1

    if not X:
        raise SystemExit(
            "No usable rows.  Add labels, or pass --pseudo-labels to "
            "bootstrap from the rule engine."
        )

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int32),
        np.asarray(groups),
        dict(stats),
    )


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def _make_model(random_state: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        class_weight="balanced",
        random_state=random_state,
    )


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (proba >= threshold).astype(np.int32)
    result: Dict[str, float] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, proba))
        result["average_precision"] = float(average_precision_score(y_true, proba))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    result["false_alarm_rate"] = float(fp / max(1, fp + tn))
    return result


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        from sklearn.metrics import f1_score

        score = f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t


def _group_split(groups: np.ndarray, test_size: float, seed: int):
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def _group_cv(groups: np.ndarray, n_splits: int, seed: int):
    from sklearn.model_selection import GroupKFold

    unique = np.unique(groups)
    n_splits = min(n_splits, len(unique))
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


def main():
    parser = argparse.ArgumentParser(
        description="Train the learned risk calibration model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--features", nargs="+", required=True,
                        help="One or more feature CSVs (or a directory)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pseudo-labels", action="store_true",
                        help="Derive weak labels from risk_level when label is blank")
    parser.add_argument("--min-water-ratio", type=float, default=0.0,
                        help="Drop negative rows below this water ratio (e.g. 0.3)")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Grouped CV folds on the training split (0 disables)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed probability threshold (default: tune on holdout)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    files = _expand_paths(args.features)
    X, y, groups, stats = load_rows(
        files, FEATURE_ORDER, args.pseudo_labels, args.min_water_ratio
    )

    counts = Counter(y.tolist())
    print(f"Loaded {len(y)} rows from {len(files)} file(s)")
    for key, value in stats.items():
        print(f"  {key:<16}: {value}")
    print(f"  positives       : {counts.get(1, 0)}")
    print(f"  negatives       : {counts.get(0, 0)}")
    print(f"  groups (tracks) : {len(np.unique(groups))}")

    if counts.get(1, 0) < 10 or counts.get(0, 0) < 10:
        print("\nWARNING: very few examples per class; the model will be weak.")
        print("Collect more labelled tracks before relying on it.\n")

    unique_groups = np.unique(groups)
    metrics: Dict[str, object] = {
        "rows": int(len(y)),
        "positives": int(counts.get(1, 0)),
        "negatives": int(counts.get(0, 0)),
        "groups": int(len(unique_groups)),
        "label_mode": "pseudo" if args.pseudo_labels else "manual",
    }

    threshold = args.threshold if args.threshold is not None else 0.5
    holdout_metrics: Optional[Dict[str, float]] = None

    if len(unique_groups) >= 4:
        train_idx, test_idx = _group_split(groups, args.test_size, args.seed)
        model = _make_model(args.seed)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]

        if args.threshold is None:
            threshold = _best_threshold(y[test_idx], proba)
        holdout_metrics = _metrics(y[test_idx], proba, threshold)
        metrics["holdout"] = holdout_metrics
        print("\nHold-out (grouped by track):")
        print(json.dumps(holdout_metrics, indent=2, sort_keys=True))

        if args.cv_folds and len(unique_groups) >= 3:
            folds = _group_cv(groups[train_idx], args.cv_folds, args.seed)
            aucs, f1s = [], []
            for train_fold, val_fold in folds:
                fold_model = _make_model(args.seed)
                fold_model.fit(X[train_idx][train_fold], y[train_idx][train_fold])
                fold_proba = fold_model.predict_proba(
                    X[train_idx][val_fold]
                )[:, 1]
                fold_metrics = _metrics(y[train_idx][val_fold], fold_proba, threshold)
                if "roc_auc" in fold_metrics:
                    aucs.append(fold_metrics["roc_auc"])
                f1s.append(fold_metrics["f1"])
            if aucs:
                metrics["cv_roc_auc_mean"] = float(np.mean(aucs))
                metrics["cv_roc_auc_std"] = float(np.std(aucs))
            metrics["cv_f1_mean"] = float(np.mean(f1s)) if f1s else 0.0
            print(f"\nGrouped CV ({len(folds)} folds): "
                  f"AUC {metrics.get('cv_roc_auc_mean', float('nan')):.3f} "
                  f"+/- {metrics.get('cv_roc_auc_std', float('nan')):.3f}, "
                  f"F1 {metrics['cv_f1_mean']:.3f}")

        # Final model sees everything.
        combined_count = counts.get(1, 0) + counts.get(0, 0)
    else:
        print("\nToo few independent tracks for a hold-out split; "
              "training on all rows.")
        combined_count = counts.get(1, 0) + counts.get(0, 0)

    if combined_count == 0:
        raise SystemExit("No labelled rows to train on.")

    final_model = _make_model(args.seed)
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "features": list(FEATURE_ORDER),
        "threshold": float(threshold),
        "metrics": metrics,
        "label_mode": metrics["label_mode"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.no_save:
        print("\n--no-save set; model not written.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(bundle, fh)

    print(f"\nSaved {output}")
    print(f"RiskEngine will blend this model automatically on next start.")


if __name__ == "__main__":
    main()
