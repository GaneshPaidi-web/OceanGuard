"""
app.py
======
AquaGuard AI — drowning detection web application.

The original Flask app only drew a frame counter on the video.  This version
runs the full detection pipeline:

    upload -> background job -> D-FINE + pose + risk engine -> annotated MP4
           -> de-duplicated alert log -> results dashboard

It also serves a live MJPEG dashboard (``/stream``) so the same pipeline can
be pointed at a beach camera, an underwater camera or a sonar feed.

Run:
    python app.py
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename

from config import load_config, pipeline_kwargs
from detection_pipeline import DetectionPipeline, PipelineConfig
from geofence import Geofence
from video_processor import VideoProcessor


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
GEOFENCE_PATH = BASE_DIR / "configs" / "geofence.json"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

CONFIG = load_config()
ONNX_PATH = CONFIG["detector"]["onnx"]
POSE_MODEL_PATH = CONFIG["detector"]["pose"]

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "m4v", "webm"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

DEFAULT_SAMPLE_FPS = float(CONFIG["processing"]["sample_fps"])
MAX_ALERTS_KEPT = 200

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", CONFIG["server"].get("secret_key", "aquaguard-development-key")
)
app.config["MAX_CONTENT_LENGTH"] = int(
    CONFIG["processing"]["max_upload_mb"]
) * 1024 * 1024


# --------------------------------------------------
# PIPELINE SINGLETON
# --------------------------------------------------

_pipeline: Optional[DetectionPipeline] = None
_pipeline_lock = threading.Lock()
_job_lock = threading.Lock()
_process_lock = threading.Lock()


def _new_pipeline() -> DetectionPipeline:
    return DetectionPipeline(config=PipelineConfig(**pipeline_kwargs(CONFIG)))


def get_pipeline() -> DetectionPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            print("[app] Loading detection pipeline…")
            _pipeline = _new_pipeline()
            print("[app] Pipeline ready.")
        return _pipeline


# --------------------------------------------------
# JOB MANAGER
# --------------------------------------------------

class Job:
    def __init__(self, job_id: str, file_path: str, file_type: str,
                 original_name: str, options: dict):
        self.id = job_id
        self.file_path = file_path
        self.file_type = file_type
        self.original_name = original_name
        self.options = options
        self.status = "queued"       # queued | running | done | error
        self.progress = 0.0
        self.message = "Queued…"
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "original_name": self.original_name,
            "file_type": self.file_type,
        }


_jobs: Dict[str, Job] = {}


def _update_job(job: Job, fraction: float, message: str):
    job.progress = max(0.0, min(1.0, float(fraction)))
    job.message = message


def _run_job(job: Job):
    try:
        with _process_lock:          # one heavy job at a time on CPU
            pipeline = get_pipeline()
            processor = VideoProcessor(pipeline)

            job.status = "running"
            _update_job(job, 0.01, "Starting pipeline…")

            def report(fraction: float, message: str):
                _update_job(job, fraction, message)

            if job.file_type == "image":
                result = processor.process_image(
                    job.file_path,
                    str(OUTPUT_FOLDER),
                    force_underwater=(
                        True if job.options.get("force_underwater") else None
                    ),
                    progress=report,
                    job_id=job.id,
                )
            else:
                result = processor.process(
                    job.file_path,
                    str(OUTPUT_FOLDER),
                    sample_fps=job.options.get("sample_fps", DEFAULT_SAMPLE_FPS),
                    max_frames=job.options.get("max_frames", 0),
                    force_underwater=(
                        True if job.options.get("force_underwater") else None
                    ),
                    progress=report,
                    job_id=job.id,
                )

            if not result.success:
                job.status = "error"
                job.error = result.error or "Processing failed."
                job.message = job.error
                return

            job.result = result.as_dict()
            job.status = "done"
            job.progress = 1.0
            job.message = "Complete"
    except Exception as exc:  # pragma: no cover - defensive
        job.status = "error"
        job.error = str(exc)
        job.message = f"Processing failed: {exc}"
        app.logger.exception("Job %s failed", job.id)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _cleanup_old_jobs(max_age_seconds: int = 6 * 3600):
    now = time.time()
    with _job_lock:
        stale = [jid for jid, job in _jobs.items()
                 if now - job.created > max_age_seconds]
        for jid in stale:
            _jobs.pop(jid, None)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("Please select a file.")
        return redirect(url_for("home"))

    file = request.files["file"]
    if not file.filename:
        flash("No file selected.")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Unsupported file type.")
        return redirect(url_for("home"))

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    file_path = UPLOAD_FOLDER / unique_name
    file.save(str(file_path))

    file_type = "image" if extension in IMAGE_EXTENSIONS else "video"

    options = {
        "force_underwater": request.form.get("force_underwater") == "1",
        "sample_fps": float(request.form.get("sample_fps", DEFAULT_SAMPLE_FPS)),
    }

    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, str(file_path), file_type, original_name, options)

    with _job_lock:
        _jobs[job_id] = job

    _cleanup_old_jobs()

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()

    return redirect(url_for("results", job_id=job_id))


@app.route("/results/<job_id>")
def results(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        flash("That analysis session has expired. Please upload again.")
        return redirect(url_for("home"))

    # Base metadata that the template can always render.
    context = {
        "job_id": job.id,
        "original_name": job.original_name,
        "filename": Path(job.file_path).name,
        "file_type": job.file_type,
        "processing": job.status in ("queued", "running"),
        "scene_type": "surface",
        "conf_threshold": "0.40",
        "width": 0,
        "height": 0,
        "fps": 0,
        "output_fps": 0,
        "frame_count": 0,
        "processed_count": 0,
        "duration": 0,
        "alerts": [],
        "drowning_count": 0,
        "distress_count": 0,
        "normal_count": 0,
        "unknown_count": 0,
        "geofence_count": 0,
        "geofence_alert_count": 0,
        "csv_name": None,
        "alerts_csv_name": None,
        "output_name": None,
        "preview_name": None,
        "stats": {},
        "error": job.error,
    }

    if job.status == "error":
        context["error"] = job.error
        return render_template("results.html", **context)

    if job.status == "done" and job.result:
        result = job.result
        stats = result.get("stats", {})
        context.update({
            "processing": False,
            "output_name": result.get("output_name"),
            "preview_name": result.get("preview_name"),
            "csv_name": result.get("csv_name"),
            "alerts_csv_name": result.get("alerts_csv_name"),
            "alerts": result.get("alerts", []),
            "stats": stats,
            "scene_type": stats.get("scene_type", "surface"),
            "conf_threshold": stats.get("conf_threshold", 0.4),
            "width": stats.get("width", 0),
            "height": stats.get("height", 0),
            "fps": stats.get("fps", 0),
            "output_fps": stats.get("output_fps", 0),
            "frame_count": stats.get("frame_count", 0),
            "processed_count": stats.get("processed_count", 0),
            "duration": stats.get("duration", 0),
            "drowning_count": stats.get("drowning_count", 0),
            "distress_count": stats.get("distress_count", 0),
            "normal_count": stats.get("normal_count", 0),
            "unknown_count": stats.get("unknown_count", 0),
            "geofence_count": stats.get("geofence_count", 0),
            "geofence_alert_count": stats.get("geofence_alert_count", 0),
            "tracks": result.get("tracks", []),
        })

    return render_template("results.html", **context)


@app.route("/job/<job_id>")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(job.to_dict())


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(str(OUTPUT_FOLDER), filename)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "pipeline_loaded": _pipeline is not None,
        "onnx": ONNX_PATH,
        "pose_model": POSE_MODEL_PATH,
        "geofence_enabled": bool(
            _pipeline is not None and _pipeline.geofence.active
        ),
        "jobs": len(_jobs),
    })


# --------------------------------------------------
# GEOFENCE CONFIGURATION API
# --------------------------------------------------

@app.route("/geofence", methods=["GET", "POST"])
def geofence_api():
    if request.method == "GET":
        pipeline = get_pipeline()
        return jsonify(pipeline.geofence.as_dict())

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Expected a JSON object"}), 400

    # Merge over the current state so the UI can send only what it changed
    # (typically "enabled" and "zones") without losing tuning values.
    pipeline = get_pipeline()
    merged = dict(pipeline.geofence.as_dict())
    merged.update(payload)

    probe = Geofence.from_config(merged, base_dir=BASE_DIR)
    if bool(merged.get("enabled", probe.enabled)) and not probe.zones:
        return jsonify({
            "ok": False,
            "error": "At least one zone with 3+ points is required",
        }), 400

    state = probe.as_dict()
    GEOFENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GEOFENCE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)

    reload_payload = {"config_file": "configs/geofence.json", "enabled": state["enabled"]}
    pipeline.set_geofence(reload_payload)
    if _streamer is not None:
        _streamer.pipeline.set_geofence(reload_payload)

    return jsonify({"ok": True, "geofence": pipeline.geofence.as_dict()})


# --------------------------------------------------
# LIVE STREAM
# --------------------------------------------------

class LiveStreamer:
    """Processes a camera / video source and serves annotated MJPEG frames."""

    def __init__(
        self,
        pipeline: DetectionPipeline,
        source: str = "",
        max_width: int = 960,
        min_process_interval: float = 0.7,
    ):
        self.pipeline = pipeline
        self.source = source
        self.max_width = max_width
        self.min_process_interval = min_process_interval
        self._frame_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._alerts: deque = deque(maxlen=MAX_ALERTS_KEPT)
        self._status = "STOPPED"
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_index = 0

    # --------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._status = "STOPPED"

    def latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._jpeg

    def alerts_snapshot(self) -> list:
        return list(self._alerts)

    def status(self) -> str:
        return self._status

    # --------------------------------------------------

    def _open_source(self):
        if self.source.isdigit():
            return cv2.VideoCapture(int(self.source))
        if self.source and Path(self.source).exists():
            return cv2.VideoCapture(self.source)
        return cv2.VideoCapture(0)

    def _loop(self):
        cap = self._open_source()
        if not cap.isOpened():
            # Last resort: loop a bundled clip so the dashboard still works.
            fallback = BASE_DIR / "test_images" / "beach.mp4"
            if fallback.exists():
                cap = cv2.VideoCapture(str(fallback))
            if not cap.isOpened():
                self._status = "NO SOURCE"
                self._render_placeholder("No camera or video source available")
                return

        self._status = "LIVE"
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_delay = 1.0 / max(1.0, min(60.0, source_fps))
        last_process = 0.0
        cached_results = []
        cached_annotated = None
        self.pipeline.reset()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                # Loop file sources.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break

            h, w = frame.shape[:2]
            if w > self.max_width:
                scale = self.max_width / float(w)
                frame = cv2.resize(frame, (self.max_width, int(h * scale)))

            now = time.perf_counter()
            if now - last_process >= self.min_process_interval:
                last_process = now
                try:
                    results = self.pipeline.process_frame(
                        frame,
                        frame_index=self._frame_index,
                        source_fps=max(1.0, source_fps),
                    )
                    cached_results = results
                    cached_annotated = self.pipeline.annotate_frame(frame, results)

                    for entry in results:
                        if entry.get("geofence_event") and entry.get("geofence_breach"):
                            self._alerts.append({
                                "timestamp": entry["timestamp"],
                                "track_id": entry["track_id"],
                                "risk_level": "GEOFENCE",
                                "alert_type": "GEOFENCE",
                                "geofence_zone": entry.get("geofence_zone"),
                                "confidence": entry["confidence"],
                                "reasons": (
                                    entry.get("geofence_reason")
                                    or "Person crossed the safety geofence"
                                ),
                            })
                        if entry["event"] and entry["risk_level"] in ("DROWNING", "DISTRESS"):
                            self._alerts.append({
                                "timestamp": entry["timestamp"],
                                "track_id": entry["track_id"],
                                "risk_level": entry["risk_level"],
                                "alert_type": "RISK",
                                "geofence_zone": None,
                                "confidence": entry["confidence"],
                                "reasons": "; ".join(entry["reasons"][:3]),
                            })
                except Exception as exc:  # pragma: no cover
                    app.logger.exception("Live processing failed: %s", exc)

            display = cached_annotated if cached_annotated is not None else frame
            if display.shape[:2] != frame.shape[:2]:
                display = cv2.resize(display, (frame.shape[1], frame.shape[0]))

            ok, buffer = cv2.imencode(
                ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 78]
            )
            if ok:
                with self._frame_lock:
                    self._jpeg = buffer.tobytes()

            self._frame_index += 1
            time.sleep(frame_delay * 0.5)

        cap.release()
        self._status = "STOPPED"

    def _render_placeholder(self, text: str):
        frame = np.zeros((360, 960, 3), dtype=np.uint8)
        cv2.putText(
            frame, text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (200, 200, 200), 2,
        )
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            with self._frame_lock:
                self._jpeg = buffer.tobytes()


_streamer: Optional[LiveStreamer] = None
_streamer_lock = threading.Lock()


def get_streamer() -> LiveStreamer:
    global _streamer
    with _streamer_lock:
        if _streamer is None:
            stream = CONFIG.get("stream", {})
            # The streamer gets its own pipeline: a live feed and a batch job
            # must not share tracker/hysteresis state.
            _streamer = LiveStreamer(
                _new_pipeline(),
                source=stream.get("source", ""),
                max_width=int(stream.get("max_width", 960)),
                min_process_interval=float(
                    stream.get("min_process_interval", 0.7)
                ),
            )
        return _streamer


@app.route("/stream")
def stream_page():
    return render_template("stream.html")


@app.route("/video_feed")
def video_feed():
    streamer = get_streamer()
    streamer.start()

    def generate():
        while True:
            jpeg = streamer.latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            time.sleep(0.08)

    return app.response_class(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream_alerts")
def stream_alerts():
    streamer = get_streamer()
    return jsonify({
        "status": streamer.status(),
        "alerts": streamer.alerts_snapshot(),
    })


@app.route("/stream_control", methods=["POST"])
def stream_control():
    streamer = get_streamer()
    action = request.form.get("action", request.json.get("action") if request.is_json else "")
    if action == "stop":
        streamer.stop()
    else:
        streamer.start()
    return jsonify({"status": streamer.status()})


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    flash("File is too large. Increase MAX_UPLOAD_MB to allow larger uploads.")
    return redirect(url_for("home"))


@app.errorhandler(500)
def internal_error(error):  # pragma: no cover
    app.logger.exception("Unhandled error")
    return render_template("results.html", error=str(error),
                           processing=False, file_type="video"), 500


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    print("------------------------------------")
    print("AQUAGUARD AI — DROWNING DETECTION")
    print("------------------------------------")
    print("Project:", BASE_DIR)
    print("D-FINE:  ", ONNX_PATH)
    print("Pose:    ", POSE_MODEL_PATH)
    print("Sample FPS:", DEFAULT_SAMPLE_FPS)
    print("------------------------------------")

    app.run(
        host=CONFIG["server"]["host"],
        port=int(CONFIG["server"]["port"]),
        debug=False,
        threaded=True,
    )
