"""
tools/download_datasets.py
==========================
Download public drowning / swimmer datasets and register them with the
project.  Nothing is downloaded unless you pass a flag.

Available sources
-----------------
1. ``--figshare-underwater``
   Underwater Drowning Detection Dataset (Alzaabi et al., 2025)
   5,613 annotated underwater images (Swimming / Struggling / Drowning),
   YOLO-format boxes, 640x640.  1.6 GB.  CC BY 4.0.
   https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235

2. ``--kaggle <slug>``
   Any Kaggle dataset, using ``kagglehub`` (``pip install kagglehub``).
   Recommended: ``alanoudawaji/swimming-and-drowning-dataset``

3. ``--roboflow WORKSPACE/PROJECT/VERSION``
   Any Roboflow Universe dataset.  Requires ``ROBOFLOW_API_KEY``.
   Useful projects:
     drowning-detection/person-detection-in-water
     university-g3h71/swimming-and-drowning-detection
     test2-nwjth/swimming-and-drowning-detection-pu5hs

After download, run::

    python tools/prepare_dataset.py --format yolo --source datasets/... --out datasets/drowning --data-yaml datasets/.../data.yaml

Note
----
The annotations are in YOLO *format* (normalised xywh).  That does **not**
mean YOLO is used for training or inference: this project fine-tunes
D-FINE, which reads the converted COCO JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"

FIGSHARE_ARTICLE = "29497235"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"


def log(message: str):
    print(f"[datasets] {message}")


def download_url(url: str, destination: Path, chunk_size: int = 1 << 20):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        log(f"Already present: {destination}")
        return

    log(f"Downloading {url}")
    log(f"       -> {destination}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(f"\r  {percent:5.1f}%  {downloaded/1e9:.2f}/{total/1e9:.2f} GB",
                      end="", flush=True)
        print()


def extract_zip(archive: Path, destination: Path):
    log(f"Extracting {archive.name} -> {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    log("Extraction complete.")


def figshare_underwater(destination: Path, keep_archive: bool):
    with urllib.request.urlopen(FIGSHARE_API) as response:
        metadata = json.load(response)

    files = metadata.get("files", [])
    if not files:
        raise RuntimeError("Figshare article has no files.")

    target = destination / "underwater_drowning"
    target.mkdir(parents=True, exist_ok=True)

    for entry in files:
        archive = target / entry["name"]
        download_url(entry["download_url"], archive)
        if archive.suffix.lower() == ".zip":
            extract_zip(archive, target)
            if not keep_archive:
                archive.unlink(missing_ok=True)

    log(f"Dataset ready at: {target}")
    log("Look for images/train, images/val and labels/{train,val}/.")
    return target


def kaggle_download(slug: str, destination: Path):
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Run: pip install kagglehub\n"
            "You also need ~/.kaggle/kaggle.json (Kaggle account API token)."
        )

    log(f"Downloading Kaggle dataset: {slug}")
    path = Path(kagglehub.dataset_download(slug))
    target = destination / slug.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)

    if path.is_dir():
        for item in path.iterdir():
            destination_item = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination_item)

    log(f"Dataset ready at: {target}")
    return target


def roboflow_download(spec: str, destination: Path, fmt: str = "yolov8"):
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "roboflow is not installed. Run: pip install roboflow"
        )

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY first (Roboflow → Settings → API Keys)."
        )

    parts = spec.strip("/").split("/")
    if len(parts) != 3:
        raise SystemExit("Expected WORKSPACE/PROJECT/VERSION, e.g. "
                         "drowning-detection/person-detection-in-water/1")

    workspace, project, version = parts
    log(f"Downloading Roboflow dataset {workspace}/{project} v{version}")

    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(int(version))
        .download(fmt, location=str(destination / project))
    )
    target = Path(dataset.location)
    log(f"Dataset ready at: {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", default=str(DATASETS_DIR))
    parser.add_argument("--figshare-underwater", action="store_true",
                        help="Download the 5,613-image underwater dataset (1.6 GB)")
    parser.add_argument("--kaggle", metavar="OWNER/SLUG")
    parser.add_argument("--roboflow", metavar="WORKSPACE/PROJECT/VERSION")
    parser.add_argument("--roboflow-format", default="yolov8")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--list", action="store_true", help="List recommended datasets")

    args = parser.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)

    if args.list or not any([args.figshare_underwater, args.kaggle, args.roboflow]):
        print("Recommended public datasets\n")
        print("  1. Figshare — Underwater Drowning Detection (5,613 images, 1.6 GB)")
        print("     python tools/download_datasets.py --figshare-underwater")
        print("     classes: Swimming / Struggling / Drowning (YOLO-format boxes)")
        print()
        print("  2. Kaggle — Swimming and Drowning DataSet")
        print("     python tools/download_datasets.py --kaggle alanoudawaji/swimming-and-drowning-dataset")
        print()
        print("  3. Roboflow Universe — Person Detection in Water (4,474 images)")
        print("     export ROBOFLOW_API_KEY=...")
        print("     python tools/download_datasets.py --roboflow drowning-detection/person-detection-in-water/1")
        print()
        print("  4. Roboflow Universe — Swimming and Drowning Detection (7,295 images)")
        print("     python tools/download_datasets.py --roboflow university-g3h71/swimming-and-drowning-detection/1")
        return

    if args.figshare_underwater:
        figshare_underwater(destination, args.keep_archive)

    if args.kaggle:
        kaggle_download(args.kaggle, destination)

    if args.roboflow:
        roboflow_download(args.roboflow, destination, args.roboflow_format)


if __name__ == "__main__":
    sys.exit(main())
