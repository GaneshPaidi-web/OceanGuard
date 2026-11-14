
import argparse
import csv
import math
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# Preserve the working detection configuration.
INPUT_SIZE = 640
TILE_W = 800
TILE_H = 640
OVERLAP_X = 0.25
OVERLAP_Y = 0.25
CONFIDENCE = 0.40
NMS_THRESHOLD = 0.50

# Tracking configuration.
CONFIRM_HITS = 3
STATIC_WINDOW_SECONDS = 5.0
STATIC_THRESHOLD_PIXELS = 10.0
LOW_MOVEMENT_SPEED = 2.0
SUDDEN_SPEED_THRESHOLD = 25.0
SUDDEN_SPEED_CHANGE = 20.0
STATIONARY_ALERT_SECONDS = 5.0


def tile_positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    stride = max(1, int(tile_size * (1 - overlap)))
    positions = list(range(0, length - tile_size + 1, stride))

    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)

    return positions


def prepare_tile(rgb_tile):
    image = Image.fromarray(rgb_tile)
    width, height = image.size

    ratio = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_w = int(width * ratio)
    new_h = int(height * ratio)

    resized = image.resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )

    pad_w = (INPUT_SIZE - new_w) // 2
    pad_h = (INPUT_SIZE - new_h) // 2

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    canvas.paste(resized, (pad_w, pad_h))

    data = np.asarray(canvas, dtype=np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))[None, ...]

    return data, ratio, pad_w, pad_h


def calculate_iou(a, b):
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])

    intersection = (
        max(0, xx2 - xx1) * max(0, yy2 - yy1)
    )

    area_a = (
        max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    )
    area_b = (
        max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def nms(detections):
    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)

        remaining = []

        for item in detections:
            if item["class_id"] != best["class_id"]:
                remaining.append(item)
                continue

            if calculate_iou(best["box"], item["box"]) < NMS_THRESHOLD:
                remaining.append(item)

        detections = remaining

    return kept


class PersonTracker:

    def __init__(self, max_age=2, iou_threshold=0.10):
        self.next_id = 1
        self.tracks = {}
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    @staticmethod
    def center(box):
        return (
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        )

    @staticmethod
    def diagonal(box):
        w = max(1, box[2] - box[0])
        h = max(1, box[3] - box[1])
        return math.hypot(w, h)

    def match_score(self, old_box, new_box):
        iou = calculate_iou(old_box, new_box)

        old_center = self.center(old_box)
        new_center = self.center(new_box)

        distance = math.dist(old_center, new_center)

        max_distance = max(
            20.0,
            self.diagonal(old_box) * 1.5
        )

        if iou < self.iou_threshold and distance > max_distance:
            return None

        distance_score = max(
            0.0, 1.0 - distance / max_distance
        )

        return 0.65 * iou + 0.35 * distance_score

    def movement_status(self, track, timestamp):
        history = track["history"]

        while (
            history
            and history[0][0] < timestamp - STATIC_WINDOW_SECONDS
        ):
            history.popleft()

        if len(history) < 2:
            track["previous_speed"] = 0.0
            track["stationary_since"] = None
            return "INSUFFICIENT_HISTORY", 0.0, 0.0, False

        previous_time, previous_center = history[-2]
        current_center = history[-1][1]

        elapsed = timestamp - previous_time
        speed = (
            math.dist(previous_center, current_center) / elapsed
            if elapsed > 0 else 0.0
        )

        previous_speed = track.get("previous_speed", 0.0)
        sudden_movement = (
            speed >= SUDDEN_SPEED_THRESHOLD
            or (
                len(history) >= 3
                and speed - previous_speed >= SUDDEN_SPEED_CHANGE
            )
        )
        track["previous_speed"] = speed

        if speed <= LOW_MOVEMENT_SPEED:
            if track.get("stationary_since") is None:
                track["stationary_since"] = previous_time
        else:
            track["stationary_since"] = None

        stationary_duration = (
            timestamp - track["stationary_since"]
            if track.get("stationary_since") is not None else 0.0
        )

        possibly_static = False
        if timestamp - history[0][0] >= STATIC_WINDOW_SECONDS:
            first_center = history[0][1]
            maximum_displacement = max(
                math.dist(first_center, point)
                for _, point in history
            )
            possibly_static = (
                maximum_displacement <= STATIC_THRESHOLD_PIXELS
            )

        if possibly_static or stationary_duration >= STATIONARY_ALERT_SECONDS:
            status = "POSSIBLY_STATIC"
        elif sudden_movement:
            status = "SUDDEN_MOVEMENT"
        elif speed > LOW_MOVEMENT_SPEED:
            status = "MOVING"
        else:
            status = "LOW_MOVEMENT"

        return status, speed, stationary_duration, sudden_movement

    def update(self, detections, timestamp):
        track_ids = list(self.tracks.keys())
        candidates = []

        for track_id in track_ids:
            old_box = self.tracks[track_id]["box"]

            for index, detection in enumerate(detections):
                score = self.match_score(
                    old_box, detection["box"]
                )

                if score is not None:
                    candidates.append((score, track_id, index))

        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()
        assignments = {}

        for score, track_id, index in candidates:
            if track_id in matched_tracks:
                continue

            if index in matched_detections:
                continue

            track = self.tracks[track_id]
            detection = detections[index]

            # A gap breaks consecutive-hit confirmation.
            if track["missed"] == 0:
                track["consecutive_hits"] += 1
            else:
                track["consecutive_hits"] = 1

            track["hits"] += 1
            track["missed"] = 0
            track["box"] = detection["box"]

            center = self.center(detection["box"])
            track["history"].append((timestamp, center))

            status, speed, stationary_duration, sudden_movement = (
                self.movement_status(track, timestamp)
            )

            if track["consecutive_hits"] >= CONFIRM_HITS:
                track["confirmed"] = True

            assignments[index] = {
                "track_id": track_id,
                "confirmed": track["confirmed"],
                "consecutive_hits": track["consecutive_hits"],
                "movement_status": status,
                "speed": speed,
                "stationary_duration": stationary_duration,
                "sudden_movement": sudden_movement
            }

            matched_tracks.add(track_id)
            matched_detections.add(index)

        # Age unmatched tracks.
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]["missed"] += 1
                self.tracks[track_id]["consecutive_hits"] = 0

        # Remove expired tracks.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_age
        ]

        for track_id in expired:
            del self.tracks[track_id]

        # Create new candidate tracks.
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            center = self.center(detection["box"])

            self.tracks[track_id] = {
                "box": detection["box"],
                "hits": 1,
                "consecutive_hits": 1,
                "missed": 0,
                "confirmed": False,
                "history": deque([(timestamp, center)]),
                "previous_speed": 0.0,
                "stationary_since": None
            }

            assignments[index] = {
                "track_id": track_id,
                "confirmed": False,
                "consecutive_hits": 1,
                "movement_status": "INSUFFICIENT_HISTORY",
                "speed": 0.0,
                "stationary_duration": 0.0,
                "sudden_movement": False
            }

        return assignments


def detect_frame(session, frame):
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    xs = tile_positions(width, TILE_W, OVERLAP_X)
    ys = tile_positions(height, TILE_H, OVERLAP_Y)

    detections = []
    tile_number = 0

    for y in ys:
        for x in xs:
            tile_number += 1

            tile = rgb[
                y:min(y + TILE_H, height),
                x:min(x + TILE_W, width)
            ]

            input_data, ratio, pad_w, pad_h = prepare_tile(tile)

            orig_sizes = np.array(
                [[INPUT_SIZE, INPUT_SIZE]], dtype=np.int64
            )

            labels, boxes, scores = session.run(
                None,
                {
                    "images": input_data,
                    "orig_target_sizes": orig_sizes
                }
            )

            for label, box, score in zip(
                labels[0], boxes[0], scores[0]
            ):
                confidence = float(score)

                if int(label) != 0 or confidence < CONFIDENCE:
                    continue

                x1 = (float(box[0]) - pad_w) / ratio + x
                y1 = (float(box[1]) - pad_h) / ratio + y
                x2 = (float(box[2]) - pad_w) / ratio + x
                y2 = (float(box[3]) - pad_h) / ratio + y

                x1 = max(0, min(width, x1))
                y1 = max(0, min(height, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({
                    "class_id": int(label),
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "tile_id": tile_number
                })

    return nms(detections), len(xs) * len(ys)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--onnx", default="models/dfine_s_coco.onnx"
    )
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-age", type=int, default=2)

    args = parser.parse_args()

    if args.sample_fps <= 0 or args.max_age < 0:
        raise ValueError("Invalid sampling or tracking settings")

    video_path = Path(args.video)
    model_path = Path(args.onnx)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        raise RuntimeError("Invalid video FPS")

    frame_step = max(1, round(source_fps / args.sample_fps))
    actual_sample_fps = source_fps / frame_step

    output_video = video_path.with_name(
        "tracked_detection_output.mp4"
    )
    output_csv = video_path.with_name(
        "tracked_detection_results.csv"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_sample_fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    tracker = PersonTracker(max_age=args.max_age)

    print("D-FINE-S loaded using CPUExecutionProvider")
    print(f"Video: {width}x{height}, {source_fps:.2f} FPS")
    print(f"Sampling: {actual_sample_fps:.2f} FPS")
    print(f"Source frames: {total_frames}")
    print("Track confirmation: 3 consecutive samples")
    print("Starting detection and tracking...")

    frame_index = 0
    sample_count = 0
    total_detections = 0
    start_time = time.perf_counter()

    try:
        with open(
            output_csv, "w", newline="", encoding="utf-8"
        ) as file:

            csv_writer = csv.writer(file)

            csv_writer.writerow([
                "frame_index",
                "timestamp_seconds",
                "track_id",
                "confirmed",
                "consecutive_hits",
                "confidence",
                "movement_status",
                "movement_speed_pixels_per_second",
                "stationary_duration_seconds",
                "sudden_movement_flag",
                "static_object_flag",
                "behaviour_review_flag",
                "drowning_analysis_status",
                "x1", "y1", "x2", "y2"
            ])

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_index % frame_step == 0:
                    timestamp = frame_index / source_fps

                    detections, tile_count = detect_frame(
                        session, frame
                    )

                    assignments = tracker.update(
                        detections, timestamp
                    )

                    for index, detection in enumerate(detections):
                        x1, y1, x2, y2 = detection["box"]
                        confidence = detection["confidence"]

                        info = assignments[index]
                        track_id = info["track_id"]
                        confirmed = info["confirmed"]
                        movement = info["movement_status"]
                        speed = info["speed"]
                        stationary_duration = info["stationary_duration"]
                        sudden_movement = info["sudden_movement"]

                        static_flag = (
                            movement == "POSSIBLY_STATIC"
                        )
                        behaviour_review = (
                            static_flag or sudden_movement
                        )

                        # Orange = candidate; green = confirmed.
                        color = (
                            (0, 165, 255)
                            if not confirmed
                            else (0, 255, 0)
                        )

                        if static_flag:
                            color = (255, 255, 0)
                        elif sudden_movement:
                            color = (0, 0, 255)

                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color, 2
                        )

                        label = (
                            f"{'Person' if confirmed else 'Candidate'} "
                            f"ID {track_id} {movement}"
                        )

                        cv2.putText(
                            frame,
                            label,
                            (int(x1), max(20, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color, 2
                        )

                        csv_writer.writerow([
                            frame_index,
                            round(timestamp, 3),
                            track_id,
                            confirmed,
                            info["consecutive_hits"],
                            round(confidence, 5),
                            movement,
                            round(speed, 2),
                            round(stationary_duration, 2),
                            sudden_movement,
                            static_flag,
                            behaviour_review,
                            "NOT_RUN",
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2)
                        ])

                    writer.write(frame)

                    sample_count += 1
                    total_detections += len(detections)

                    if sample_count % 3 == 0:
                        elapsed = time.perf_counter() - start_time
                        confirmed_count = sum(
                            1 for t in tracker.tracks.values()
                            if t["confirmed"]
                        )

                        print(
                            f"Samples: {sample_count}, "
                            f"tiles/frame: {tile_count}, "
                            f"detections: {total_detections}, "
                            f"active confirmed tracks: {confirmed_count}, "
                            f"elapsed: {elapsed:.1f}s"
                        )

                    if (
                        args.max_samples > 0
                        and sample_count >= args.max_samples
                    ):
                        break

                frame_index += 1

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start_time

    print("\nTracking completed.")
    print(f"Processed samples: {sample_count}")
    print(f"Total detections: {total_detections}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Output video: {output_video}")
    print(f"Output CSV: {output_csv}")
    print("Movement analysis: completed")
    print("Behaviour review flags: sudden movement / prolonged low movement")
    print("Note: these flags are indicators for review, not drowning diagnoses.")


if __name__ == "__main__":
    main()
