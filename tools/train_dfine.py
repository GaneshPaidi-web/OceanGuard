"""
tools/train_dfine.py
====================
Thin launcher around D-FINE's ``train.py`` with sensible defaults for the
drowning dataset.

Examples
--------
Fine-tune D-FINE-S from the COCO checkpoint on a GPU::

    python tools/train_dfine.py \
        --config D-FINE/configs/dfine/custom/dfine_hgnetv2_s_drowning.yml \
        --tuning D-FINE/dfine_s_coco.pth \
        --epochs 60 --batch-size 16

Evaluate only::

    python tools/train_dfine.py --config ... --resume output/.../best.pth --test

Export an existing checkpoint to ONNX (does not retrain)::

    python tools/train_dfine.py --config ... --resume output/.../best.pth --export

Train and then export::

    python tools/train_dfine.py --config ... --resume output/.../last.pth \
        --epochs 20 --train-first --export

Notes
-----
* Training D-FINE on a CPU is possible but extremely slow (hours per
  epoch for a few thousand images).  Use a CUDA GPU or Colab/Kaggle.
* The default recipe freezes nothing and uses the standard D-FINE
  augmentations.  For small datasets (a few thousand images) 40-80 epochs
  is usually enough; watch ``val`` AP and stop when it plateaus.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DFINE_DIR = PROJECT_ROOT / "D-FINE"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "dfine_s_coco.pth"


def log(message: str):
    print(f"[train] {message}")


def find_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuning", default=str(DEFAULT_WEIGHTS),
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--resume", help="Resume / evaluate this checkpoint")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override epochs from the config")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Override total_batch_size")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="Evaluation only")
    parser.add_argument("--export", action="store_true",
                        help="Export the --resume checkpoint to ONNX")
    parser.add_argument("--train-first", action="store_true",
                        help="With --export, run training before exporting")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Extra args passed to train.py")
    args = parser.parse_args()

    if args.export and not args.resume:
        raise SystemExit("--export requires --resume <checkpoint.pth>")

    config = Path(args.config)
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")

    if not (DFINE_DIR / "train.py").exists():
        raise SystemExit(f"D-FINE checkout not found at {DFINE_DIR}")

    cuda = find_cuda()
    device = args.device or ("cuda:0" if cuda else "cpu")

    if device == "cpu" and not args.test and not args.export:
        log("WARNING: CUDA is not available. D-FINE training on CPU is")
        log("         extremely slow. Strongly consider a GPU runtime.")

    command = [
        sys.executable, "train.py",
        "-c", str(config),
        "--device", device,
        "--seed", str(args.seed),
    ]

    if args.test:
        command.append("--test-only")
    if args.resume:
        command += ["-r", str(args.resume)]
    elif args.tuning and Path(args.tuning).exists() and not args.export:
        command += ["-t", str(args.tuning)]
    elif args.tuning:
        log(f"Pretrained checkpoint not found ({args.tuning}); training from scratch.")

    updates = []
    if args.epochs > 0:
        updates += ["epochs", str(args.epochs)]
    if args.batch_size > 0:
        updates += ["train_dataloader.total_batch_size", str(args.batch_size)]
        updates += ["val_dataloader.total_batch_size", str(args.batch_size * 2)]
    if updates:
        command += ["-u"] + updates

    if cuda and not args.test and not args.export:
        command.append("--use-amp")

    if args.output_dir:
        command += ["--output-dir", args.output_dir]

    if args.extra:
        command += args.extra

    run_training = (not args.export) or args.train_first or args.test
    if run_training:
        log("Running: " + " ".join(command))
        result = subprocess.run(command, cwd=str(DFINE_DIR))
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    if args.export:
        if not args.resume:
            raise SystemExit("--export requires --resume <checkpoint.pth>")

        export_script = DFINE_DIR / "tools" / "deployment" / "export_onnx.py"
        resume_path = Path(args.resume).resolve()
        # export_onnx.py derives the output name from the checkpoint path
        # (``resume.replace('.pth', '.onnx')``); it has no --output-file flag.
        export_onnx = resume_path.with_suffix(".onnx")
        output_onnx = PROJECT_ROOT / "models" / (
            resume_path.stem + "_drowning.onnx"
        )
        export_cmd = [
            sys.executable, str(export_script),
            "-c", str(config),
            "-r", str(resume_path),
        ]
        log("Exporting ONNX: " + " ".join(export_cmd))
        subprocess.run(export_cmd, cwd=str(DFINE_DIR), check=True)

        if not export_onnx.exists():
            raise SystemExit(
                f"Export finished but {export_onnx} was not created"
            )
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_onnx, output_onnx)

        # Large exports keep weights in a sidecar (e.g. ``model.onnx.data``).
        for sidecar in export_onnx.parent.glob(export_onnx.name + ".*"):
            shutil.copy2(sidecar, output_onnx.parent / sidecar.name)
            log(f"Copied sidecar: {sidecar.name}")

        log(f"ONNX model: {output_onnx}")
        log("Use it with: DFINE_ONNX=<path> python app.py")


if __name__ == "__main__":
    main()
