# AquaGuard AI — Drowning Detection

Temporal drowning-risk detection for **beach / surface cameras**, **underwater
cameras** and **sonar or monochrome feeds**.

The system finds people with a fine-tunable D-FINE detector, measures pose with
MediaPipe, and decides whether someone is drowning with a **water-gated,
temporal risk engine**. It is deliberately *not* a per-frame "is this person
horizontal?" classifier — that approach produces constant false alarms on
swimmers.

> **No YOLO is used for detection or inference.** Annotations may arrive in
> YOLO *format* (normalised boxes), but they are converted to COCO JSON and
> used to fine-tune D-FINE.

---

## What changed vs. the original code

| Area | Before | Now |
| --- | --- | --- |
| Decision logic | Per-frame angle thresholds | 9-second motion history + pose, noisy-OR cue fusion, EMA + hysteresis |
| Posture convention | Upright people reported as ~0° (horizontal) → false alarms | 0° = horizontal, 90° = upright, verified by tests |
| False alarms on sand | No water context | HSV water mask + per-person in-water ratio gates aquatic cues |
| Sonar / grayscale | COCO detector only (poor recall) | Classical blob detector fused with D-FINE |
| Underwater footage | One global colour cast correction | Scene classification (`surface` / `underwater` / `grayscale` / `sonar`) + dark-channel dehazing |
| Tracking | Nearest box / none | Constant-velocity tracker, track confirmation, pose caching |
| Large 4K frames | Missed small people | Automatic tiled inference + optional flip TTA |
| Web app | Drew a frame counter only | Background jobs, progress polling, annotated MP4, alert CSV, live MJPEG dashboard |
| Evaluation | None | Detector AP@0.5 and risk precision/recall/F1 tools |
| Camera motion | Not compensated | Per-frame phase-correlation compensation subtracted from track speed/sink/horizontal cues; eliminates false alerts on panning/zooming cameras |
| Rip-current geofence | None | Per-camera safe/danger polygons; person-only crossing alerts, boat/surfboard passengers excluded |

---

## Architecture

```
video / camera
      │
      ▼
underwater_preprocessor.py     scene detection + enhancement + sonar flag
      │
      ├──────────────► water_mask.py        per-person in-water ratio
      ▼
detection_pipeline.py
  ├─ D-FINE-S ONNX (tiled, batched, optional flip TTA)
  ├─ sonar_detector.py (blob detection + fusion on sonar frames)
  ├─ PersonTracker (constant velocity, confirmation, motion features)
  └─ pose scheduling (largest in-water tracks only)
      │
      ▼
drowning_classifier.py ── PoseAnalyzer (MediaPipe torso angle, head/hip,
      │                    visibility, arm asymmetry)
      ▼
risk_engine.py ── scale-invariant motion + pose cues, noisy-OR, EMA,
      │            hysteresis, water gating, optional learned calibration
      ▼
NORMAL / DISTRESS / DROWNING + reasons
      │
      ├─ result CSV + alert CSV + annotated MP4 (video_processor.py)
      └─ Flask dashboard (app.py)
```

### Why the risk engine is more accurate

1. **Scale invariance** — speeds are measured in body lengths per second, so
   one threshold works for a 20 px swimmer and a 400 px swimmer.
2. **Multiple weak cues** — inactivity, thrashing without progress, vertical
   bobbing, sinking, head submersion, pose loss, arm asymmetry and
   horizontal posture without forward progress are combined, not OR-ed from a
   single angle.
3. **Water gating** — aquatic cues are near-disabled on land (`0.12×`), while
   underwater/sonar scenes force `water_ratio = 1.0`.
4. **Temporal smoothing + hysteresis** — a level must be sustained
   (1.1 s for DROWNING) and must clear for 2.2 s before it is dropped, so
   single-frame pose noise cannot raise or clear an alert.
5. **Optional learned calibration** — `models/risk_model.pkl` can blend a
   gradient-boosted model trained on labelled tracks (see below).

---

## Geofencing — rip-current early warning

A rip current pulls swimmers away from the beach. Waiting for thrashing or
submersion is late; the geofence warns as soon as a **person** leaves the
guarded area.

Draw the safe swimming area (or an explicit danger / rip-current channel) once
for the camera view on the **Live Stream** page:

1. Start the stream so the camera view is visible.
2. Click **🟢 Draw safe zone** (or **🔴 Draw danger zone**) and click points
   around the area. **Undo point** removes the last click.
3. Tick **Enabled** and press **💾 Save geofence**.

Points are stored as **normalised coordinates** in `configs/geofence.json`, so
the same polygon works on a 720p preview and a 4K analysed frame. You can also
edit that file by hand:

```json
{
  "enabled": true,
  "anchor": "feet",
  "hold_seconds": 1.5,
  "clear_seconds": 2.5,
  "require_in_water": true,
  "exclude_platforms": true,
  "zones": [
    {"name": "guarded-beach", "mode": "safe",
     "points": [[0.03, 0.42], [0.97, 0.42], [0.97, 0.62], [0.03, 0.62]]},
    {"name": "rip-channel", "mode": "danger",
     "points": [[0.42, 0.20], [0.58, 0.20], [0.62, 0.45], [0.38, 0.45]]}
  ]
}
```

Behaviour:

| Rule | Meaning |
| --- | --- |
| `mode: safe` | Breach when the person is **outside every safe polygon** |
| `mode: danger` | Breach when the person is **inside the danger polygon** (always wins) |
| `anchor: feet` | Uses the bottom-centre (waterline) of the box; `center` uses the box centre |
| `hold_seconds` | Must stay outside/inside this long before alerting (default 1.5 s) |
| `clear_seconds` | Alert stays up this long after the person returns (hysteresis) |
| `require_in_water` | People on dry sand are ignored; `water_ratio` must pass `min_water_ratio` |
| `exclude_platforms` | People on boats / surfboards are ignored, with a short grace period |

**Person-only by design.** D-FINE's COCO class 8 (boat) and class 37
(surfboard) boxes are used only to exclude passengers — they are never tracked
or geofenced. Someone who steps off a boat into the water becomes a normal
geofence candidate after the grace period.

Geofence alerts are independent of the drowning risk level: `risk_level` stays
`NORMAL / DISTRESS / DROWNING`, while the result entry gains
`geofence_breach`, `geofence_event`, `geofence_zone` and `on_platform`. The
annotated video draws the polygons, a waterline marker and a
`GEOFENCE BREACH — possible rip current` banner. Alerts appear in the alert
CSV with `alert_type = GEOFENCE`.

API endpoints:

| Route | Purpose |
| --- | --- |
| `GET /geofence` | current zones and settings |
| `POST /geofence` | save zones (`{"enabled": true, "zones": [...]}`) and reload both pipelines |

> Camera must be fixed (or motion-compensated). A geofence drawn for one camera
> angle does not transfer to another. Boat/surfboard exclusion is best-effort
> and depends on the detector seeing the platform; mark marinas or moored boats
> with a `danger`/ignore area if they cause repeated alerts.

---

## Requirements

- Python 3.10+ (developed on 3.12)
- ~1 GB RAM for CPU inference (more for 4K tiled inference)
- Optional NVIDIA GPU for D-FINE fine-tuning; inference runs on CPU

Runtime models expected in `models/`:

| File | Purpose |
| --- | --- |
| `dfine_s_coco.onnx` | D-FINE-S person detector |
| `pose_landmarker_lite.task` | MediaPipe pose |
| `risk_model.pkl` | *optional* learned risk calibration |

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For NVIDIA inference, replace `onnxruntime` with `onnxruntime-gpu`.
`ffmpeg` is optional but strongly recommended for H.264 output; the processor
falls back to `mp4v`. Set `FFMPEG_PATH` if it is not on `PATH`.

---

## Quick start

### Web dashboard

```powershell
python app.py
# open http://127.0.0.1:5000
```

The first start takes ~30–45 s because MediaPipe/TensorFlow and ONNX Runtime are
imported eagerly; the pipeline itself loads on the first job. (Pass `-u` to
Python if you want unbuffered startup logs.)

Upload an image or video, pick the sampling rate, and watch the job progress.
The result page shows the annotated video, alerts, detection CSV, alert CSV
and a per-track summary.

Routes:

| Route | Purpose |
| --- | --- |
| `/` | upload / analysis |
| `/results/<job_id>` | job dashboard |
| `/job/<job_id>` | JSON progress |
| `/stream` | live camera dashboard |
| `/video_feed` | MJPEG stream |
| `/stream_alerts` | JSON live alerts |
| `/stream_control` | start/stop the streamer |
| `/health` | liveness + loaded model paths |

### Command line

```powershell
# Video -> annotated MP4 + CSVs
python video_processor.py test_images/beach.mp4 --output-dir outputs --sample-fps 2

# Single image
python video_processor.py path\to\image.jpg --output-dir outputs

# Underwater / sonar footage
python video_processor.py sonar.mp4 --force-underwater --sample-fps 2

# Collect features for risk-model training
python video_processor.py beach.mp4 --dump-features outputs/features/beach.csv
```

| Flag | Meaning |
| --- | --- |
| `--config` | YAML config (default `configs/pipeline.yaml`) |
| `--sample-fps` | Frames analysed per second (2 is usually enough) |
| `--max-frames` | Stop after N analysed frames |
| `--tta` | Horizontal-flip test-time augmentation (slower, more recall) |
| `--batch-size` | Tiles per ONNX call |
| `--force-underwater` | Skip scene detection |
| `--dump-features` | Per-detection features for calibration/training |

Outputs are written to `outputs/`:

- `<id>_<name>_processed.mp4` — annotated video
- `<id>_<name>_preview.jpg` — first frame
- `<id>_<name>_detections.csv` — every detection, every feature
- `<id>_<name>_alerts.csv` — de-duplicated risk transitions

---

## Configuration

Defaults live in `configs/pipeline.yaml` and are loaded by `config.py`.
Environment variables override the file:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DFINE_ONNX` | `models/dfine_s_coco.onnx` | detector path |
| `POSE_MODEL` | `models/pose_landmarker_lite.task` | pose model |
| `SAMPLE_FPS` | `2` | analysis frame rate |
| `BATCH_SIZE` | `8` | ONNX tile batch |
| `ENABLE_TTA` | `0` | flip TTA |
| `STREAM_SOURCE` | `""` | RTSP URL, video path or camera index |
| `MAX_UPLOAD_MB` | `500` | upload limit |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | server bind |
| `FFMPEG_PATH` | auto | explicit ffmpeg binary |
| `AQUAGUARD_CONFIG` | `configs/pipeline.yaml` | alternate config file |

Example: point the live dashboard at a beach camera:

```powershell
$env:STREAM_SOURCE = "rtsp://user:pass@192.168.1.20:554/stream1"
python app.py
```

---

## Improving accuracy with your own data

### 1. Get a dataset

```powershell
python tools/download_datasets.py --list
python tools/download_datasets.py --figshare-underwater
```

The Figshare set is 5,613 underwater images (Swimming / Struggling / Drowning,
CC BY 4.0). Kaggle and Roboflow sources are also supported; API keys are read
from the environment.

### 2. Convert annotations to COCO and generate a D-FINE config

```powershell
python tools/prepare_dataset.py `
  --format yolo `
  --source datasets/underwater_drowning `
  --out datasets/drowning `
  --data-yaml datasets/underwater_drowning/data.yaml
```

This writes `datasets/drowning/` plus:

- `D-FINE/configs/dataset/drowning_detection.yml`
- `D-FINE/configs/dfine/custom/dfine_hgnetv2_s_drowning.yml`

By default all classes are merged into a single `person` class, which is what
the runtime needs. Use `--class-mode keep` to preserve source classes.

### 3. Fine-tune D-FINE (GPU strongly recommended)

```powershell
python tools/train_dfine.py `
  --config D-FINE/configs/dfine/custom/dfine_hgnetv2_s_drowning.yml `
  --tuning D-FINE/dfine_s_coco.pth `
  --epochs 60 --batch-size 16
```

Then export ONNX and use it:

```powershell
python tools/train_dfine.py `
  --config D-FINE/configs/dfine/custom/dfine_hgnetv2_s_drowning.yml `
  --resume output/dfine_hgnetv2_s_drowning/best.pth `
  --export

$env:DFINE_ONNX = "models\best_drowning.onnx"
python app.py
```

CPU fine-tuning is possible but impractical (hours per epoch). Use a CUDA GPU,
Colab or Kaggle. D-FINE's own dependencies are listed in
`D-FINE/requirements.txt`.

### 4. Calibrate the risk engine (optional)

The rule engine works out of the box. To blend in a learned model:

1. Run videos with `--dump-features`.
2. Fill the `label` column: `1` = drowning/struggling, `0` = safe.
   Leave rows blank to skip them.
3. Train:

```powershell
python tools/train_risk_model.py --features outputs/features/
```

The script groups rows by video+track, reports a grouped hold-out and grouped
cross-validation, and writes `models/risk_model.pkl`. `RiskEngine` loads it on
the next start and blends `0.6 × rules + 0.4 × model`.

If you have no labels yet, `--pseudo-labels` bootstraps from the rule engine's
own levels. That is self-distillation (it smooths the rules) and is a sanity
check, not a substitute for annotations.

---

## Evaluation

Detector (COCO JSON or YOLO-format labels):

```powershell
python tools/evaluate.py detector `
  --images datasets/drowning/images/val `
  --annotations datasets/drowning/annotations/instances_val.json `
  --onnx models/dfine_s_drowning.onnx `
  --conf-threshold 0.35
```

Reports precision, recall, F1, AP@0.5 and TP/FP/FN. Add `--export metrics.json`
to save the numbers.

Risk engine (labelled feature dumps):

```powershell
python tools/evaluate.py risk --features outputs/features/ --export risk.json
```

Reports per-track precision/recall/F1, frame-level metrics and mean/median
alert latency (time from first labelled positive frame to first DROWNING).

---

## Tests

```powershell
python -m unittest discover -s tests -v
```

Covers the risk-engine behaviours (swimmer, drowning, sunbather, underwater,
event de-duplication, reset), geofence geometry / water gating / boat
exclusion / hold-clear timing, water segmentation, sonar blob detection and
detection fusion.

---

## Scene handling

| Scene | Detection | Water handling | Pose |
| --- | --- | --- | --- |
| Beach / surface | D-FINE on enhanced frame | HSV water mask, per-person lower-body ratio | MediaPipe on largest in-water tracks |
| Underwater | D-FINE + dark-channel dehazing | forced `water_ratio = 1.0` | MediaPipe |
| Grayscale / low colour | D-FINE + sonar detector | unknown → generic cues only | MediaPipe when pose is found |
| Sonar / scan converter | sonar blob detector fused with D-FINE | forced `water_ratio = 1.0` | skipped for sonar boxes |

The preprocessor uses a bottom-band water check so aerial beach footage with a
blue sky is not mistaken for underwater footage.

---

## Performance notes

- `sample_fps=2` is the main speed knob; drowning unfolds over seconds.
- 1080p on CPU: roughly 1–2 s per analysed frame depending on tiling.
- 4K frames are tiled automatically; expect ~10–12 tiles per frame.
- `--batch-size` trades memory for throughput.
- Flip TTA roughly doubles inference cost.
- Pose runs only on the largest in-water tracks (`max_pose_tracks`, default 8).
- MediaPipe may print a harmless `TypeError: 'NoneType' object is not callable`
  at interpreter shutdown on Windows; processing is unaffected.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `D-FINE ONNX model not found` | Place `dfine_s_coco.onnx` in `models/` or set `DFINE_ONNX` |
| MediaPipe model error | Place `pose_landmarker_lite.task` in `models/` or set `POSE_MODEL` |
| Video has no audio | Expected: annotated output is video-only |
| H.264 encode fails | Install ffmpeg or set `FFMPEG_PATH`; mp4v fallback is automatic |
| Too many false alerts on land | Check `water_ratio` in the detection CSV; raise `IN_WATER_THRESHOLD` in `risk_engine.py` or fix water mask thresholds | If alerts persist on beach footage, verify camera-motion compensation is working (the pipeline estimates global panning/zooming motion via `cv2.phaseCorrelate` and subtracts it from track features). Set `AQUAGUARD_DEBUG=1` to dump per-track features CSV for inspection. |
| Missed small swimmers | Increase `--tta`, raise input resolution/tiling, or fine-tune D-FINE on in-water people |
| Slow 4K processing | Lower `--sample-fps`, reduce `--batch-size` if RAM-bound, process 1080p proxies |

---

## Project layout

```
app.py                      Flask app, jobs, live stream
config.py                   YAML + env configuration
detection_pipeline.py       detector + tracker + pose scheduling + fusion
drowning_classifier.py      MediaPipe pose features + classifier wrapper
risk_engine.py              temporal risk scoring
geofence.py                 safe/danger polygons, boat exclusion, rip-current alerts
water_mask.py               water segmentation / in-water ratio
sonar_detector.py           classical sonar blobs + fusion
underwater_preprocessor.py  scene detection and enhancement
video_processor.py          end-to-end video/image processing + CLI
configs/pipeline.yaml       runtime configuration
configs/geofence.json       geofence polygons/settings for the current camera
models/                     ONNX, pose and optional risk models
templates/, static/         dashboard UI
tools/download_datasets.py  dataset downloaders
tools/prepare_dataset.py    YOLO/VOC/COCO -> COCO + D-FINE configs
tools/train_dfine.py        D-FINE training / export launcher
tools/train_risk_model.py   learned risk calibration
tools/evaluate.py           detector and risk metrics
tests/                      unit tests
```

Legacy scripts (`long_range_detection_fixed.py`, `tiled_video_detection*.py`,
`video_detection_sampled.py`, `person_detection.py`, `mediapipe_pose_test.py`)
are kept for reference and are not used by the app.

---

## Limitations and safety

This is a decision-support tool, not a replacement for lifeguards. It can miss
drowning (occlusion, poor visibility, unusual behaviour), and it can produce
false alarms (splashing, playing, unusual camera angles). Always combine it
with human supervision, and validate on footage from your own camera before
relying on it operationally.
