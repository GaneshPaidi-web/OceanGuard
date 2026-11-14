"""Sampled-frame YOLO person detection with ByteTrack for the drowning-detection project.

Run from D:\\Drowning_Detection:
    python video_detection_sampled.py

This is a person-detection/tracking test only. It does NOT classify drowning.
"""

from pathlib import Path
import csv
import time

import cv2
from ultralytics import YOLO


# --------------------------- SETTINGS ---------------------------
VIDEO_PATH = Path("test_images/person_test.mp4")
MODEL_PATH = "yolo11n.pt"          # Try yolo11s.pt later only if needed
TARGET_FPS = 5.0                   # Process about 5 frames per second
IMAGE_SIZE = 960                   # Try 1280 later if small people are still missed
CONFIDENCE = 0.10                  # Lower values can increase false detections
MAX_INPUT_SIDE = 1920               # Downscale only if a source frame is larger
TRACKER = "bytetrack.yaml"

OUTPUT_DIR = Path("outputs")
OUTPUT_VIDEO = OUTPUT_DIR / "yolo_bytetrack_sampled.mp4"
OUTPUT_CSV = OUTPUT_DIR / "yolo_track_data.csv"


# ---------------------- HELPER FUNCTIONS ------------------------
def resize_keep_aspect(frame, max_side=MAX_INPUT_SIDE):
    """Shrink large frames without changing their aspect ratio; never upscale."""
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame

    scale = max_side / float(longest_side)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def main():
    if not VIDEO_PATH.is_file():
        print(f"Video not found: {VIDEO_PATH}")
        print("Place your video at test_images\\person_test.mp4 or update VIDEO_PATH.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(source_fps / TARGET_FPS))
    actual_sample_fps = source_fps / frame_interval

    print(f"Source FPS: {source_fps:.2f}")
    print(f"Sampling every {frame_interval} frame(s): about {actual_sample_fps:.2f} FPS")
    print(f"YOLO image size: {IMAGE_SIZE}")
    print("Starting person detection + ByteTrack...\n")

    writer = None
    frame_index = 0
    processed_frames = 0
    total_detections = 0
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

                frame = resize_keep_aspect(frame)
                frame_height, frame_width = frame.shape[:2]

                # persist=True keeps ByteTrack identities across sampled frames.
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    classes=[0],              # COCO class 0 = person
                    device="cpu",
                    verbose=False,
                )
                result = results[0]
                annotated = result.plot()

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(OUTPUT_VIDEO), fourcc, actual_sample_fps,
                        (frame_width, frame_height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not create output video. Check OpenCV video codec support.")

                boxes = result.boxes
                people_count = 0
                if boxes is not None and len(boxes) > 0:
                    people_count = len(boxes)
                    total_detections += people_count
                    coords = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    track_ids = (
                        boxes.id.int().cpu().tolist()
                        if boxes.id is not None
                        else [None] * people_count
                    )

                    for (x1, y1, x2, y2), confidence, track_id in zip(coords, confidences, track_ids):
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        box_width = x2 - x1
                        box_height = y2 - y1
                        timestamp = frame_index / source_fps
                        csv_writer.writerow([
                            f"{timestamp:.3f}", frame_index, track_id,
                            f"{confidence:.4f}", round(x1, 2), round(y1, 2),
                            round(x2, 2), round(y2, 2), round(center_x, 2),
                            round(center_y, 2), round(box_width, 2), round(box_height, 2)
                        ])

                timestamp = frame_index / source_fps
                cv2.putText(
                    annotated,
                    f"Time: {timestamp:.1f}s | People: {people_count} | Sampled: {actual_sample_fps:.1f} FPS",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                processed_frames += 1

                if processed_frames % 10 == 0:
                    print(
                        f"Processed {processed_frames} sampled frames | "
                        f"Current people: {people_count} | Source frame: {frame_index}"
                    )

                frame_index += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.time() - started
    print("\nProcessing finished.")
    print(f"Sampled frames processed: {processed_frames}")
    print(f"Total person detections across sampled frames: {total_detections}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"Track data CSV: {OUTPUT_CSV}")
    print("Note: track IDs and detection data are inputs for later behaviour analysis, not a drowning decision.")


if __name__ == "__main__":
    main()
