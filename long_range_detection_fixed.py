"""
Long-range person detection test using SAHI sliced inference + ByteTrack.

Run from D:\\Drowning_Detection:
    python long_range_detection.py

This is a detection/tracking experiment, NOT a drowning classifier.
It processes the original-resolution frame in overlapping 640x640 tiles.
"""

from pathlib import Path
import csv
import time

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# --------------------------- SETTINGS ---------------------------
VIDEO_PATH = Path("test_images/person_test.mp4")

# Start with nano for CPU speed. If it misses tiny people, test yolo11s.pt,
# but expect it to be slower. Ultralytics downloads the weights if missing.
MODEL_PATH = "yolo11n.pt"

TARGET_FPS = 2.0             # Start at 2 FPS on CPU; increase to 5 after testing
TILE_SIZE = 640               # 640x640 crops let small people use more model pixels
OVERLAP = 0.20                # 20% overlap reduces missed people on tile boundaries
CONFIDENCE = 0.10             # Low threshold helps weak distant detections; may add false positives
NMS_THRESHOLD = 0.50
TRACK_BUFFER = 30              # ByteTrack lost-track buffer, measured in sampled frames
DEVICE = "cpu"

OUTPUT_DIR = Path("outputs")
OUTPUT_VIDEO = OUTPUT_DIR / "long_range_sahi_bytetrack.mp4"
OUTPUT_CSV = OUTPUT_DIR / "long_range_track_data.csv"


def main():
    if not VIDEO_PATH.is_file():
        print(f"Video not found: {VIDEO_PATH}")
        print("Put your original video at test_images\\person_test.mp4")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model through SAHI...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE,
        device=DEVICE,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 25.0

    frame_interval = max(1, round(source_fps / TARGET_FPS))
    sampled_fps = source_fps / frame_interval
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ByteTrack receives one merged set of full-frame detections per sampled frame.
    tracker = ByteTrackTracker(
        track_activation_threshold=CONFIDENCE,
        lost_track_buffer=TRACK_BUFFER,
        minimum_consecutive_frames=1,
    )

    print(f"Video FPS: {source_fps:.2f}")
    print(f"Sampling about {sampled_fps:.2f} FPS (every {frame_interval} frames)")
    print(f"Original frame size will be retained; tile size: {TILE_SIZE}")
    print(f"Tile overlap: {OVERLAP:.0%}; device: {DEVICE}")
    print("Starting sliced person detection + ByteTrack...")
    print("Note: tiled inference is more computationally expensive than full-frame YOLO.\n")

    writer = None
    frame_index = 0
    sampled_count = 0
    total_person_detections = 0
    started = time.time()

    try:
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "timestamp_seconds", "source_frame", "track_id", "confidence",
                "x1", "y1", "x2", "y2", "center_x", "center_y", "box_width", "box_height"
            ])

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % frame_interval != 0:
                    frame_index += 1
                    continue

                height, width = frame.shape[:2]

                # SAHI splits this original-resolution frame into overlapping tiles,
                # detects in each tile, then merges overlapping detections.
                sliced_result = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=TILE_SIZE,
                    slice_width=TILE_SIZE,
                    overlap_height_ratio=OVERLAP,
                    overlap_width_ratio=OVERLAP,
                    postprocess_type="NMS",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=NMS_THRESHOLD,
                    verbose=0,
                )

                xyxy_list = []
                confidence_list = []
                class_list = []

                for prediction in sliced_result.object_prediction_list:
                    # COCO person class is 0. Ignore all other object categories.
                    if int(prediction.category.id) != 0:
                        continue
                    x1, y1, x2, y2 = prediction.bbox.to_xyxy()
                    xyxy_list.append([x1, y1, x2, y2])
                    confidence_list.append(float(prediction.score.value))
                    class_list.append(0)

                if xyxy_list:
                    detections = sv.Detections(
                        xyxy=np.asarray(xyxy_list, dtype=np.float32),
                        confidence=np.asarray(confidence_list, dtype=np.float32),
                        class_id=np.asarray(class_list, dtype=int),
                    )
                    # A final NMS pass removes any remaining duplicate tile boxes.
                    detections = detections.with_nms(threshold=NMS_THRESHOLD)
                else:
                    detections = sv.Detections.empty()

                tracked = tracker.update(detections)
                annotated = frame.copy()
                timestamp = frame_index / source_fps
                people_count = len(tracked)
                total_person_detections += people_count

                for i, (x1, y1, x2, y2) in enumerate(tracked.xyxy):
                    track_id = (
                        int(tracked.tracker_id[i])
                        if tracked.tracker_id is not None and tracked.tracker_id[i] is not None
                        else -1
                    )
                    confidence = (
                        float(tracked.confidence[i])
                        if tracked.confidence is not None
                        else 0.0
                    )

                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    label = f"Person ID {track_id} ({confidence:.2f})"
                    cv2.putText(
                        annotated, label, (x1i, max(20, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
                    )

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    csv_writer.writerow([
                        f"{timestamp:.3f}", frame_index, track_id,
                        f"{confidence:.4f}", round(float(x1), 2), round(float(y1), 2),
                        round(float(x2), 2), round(float(y2), 2),
                        round(float(center_x), 2), round(float(center_y), 2),
                        round(float(x2 - x1), 2), round(float(y2 - y1), 2)
                    ])

                cv2.putText(
                    annotated,
                    f"Time: {timestamp:.1f}s | Tracked people: {people_count} | Tiled",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                )

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(OUTPUT_VIDEO), fourcc, sampled_fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not create output video. Check OpenCV video codec support.")

                writer.write(annotated)
                sampled_count += 1

                if sampled_count % 5 == 0:
                    elapsed = time.time() - started
                    print(
                        f"Processed {sampled_count} sampled frames | "
                        f"Tracked people: {people_count} | Elapsed: {elapsed:.1f}s"
                    )

                frame_index += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.time() - started
    print("\nFinished.")
    print(f"Sampled frames: {sampled_count}")
    print(f"Total tracked detections across sampled frames: {total_person_detections}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"Tracking CSV: {OUTPUT_CSV}")
    print("\nIf distant people are still missed, we must inspect their pixel size and source video quality.")
    print("This program does not determine whether a person is drowning.")


if __name__ == "__main__":
    main()
