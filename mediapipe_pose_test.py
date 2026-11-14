
import cv2
import time
import argparse
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe pose landmark connections (33 landmarks)
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    model_path = "models/pose_landmarker_lite.task"

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("ERROR: Cannot open video:", args.video)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25

    output_path = "test_images/mediapipe_pose_output.mp4"

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Cannot create output video")
        cap.release()
        return

    frame_index = 0
    detected_frames = 0
    total_poses = 0
    start_time = time.time()

    print("MediaPipe Pose Landmarker started")
    print("Input:", args.video)
    print("Resolution:", width, "x", height)
    print("FPS:", fps)

    try:
        with vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    break

                timestamp_ms = int(frame_index * 1000 / fps)

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                pose_count = len(result.pose_landmarks)

                if pose_count > 0:
                    detected_frames += 1
                    total_poses += pose_count

                for pose in result.pose_landmarks:
                    points = []

                    for landmark in pose:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 255, 0),
                                -1
                            )

                    for start_idx, end_idx in POSE_CONNECTIONS:
                        if (
                            start_idx < len(points)
                            and end_idx < len(points)
                        ):
                            p1 = points[start_idx]
                            p2 = points[end_idx]

                            if (
                                0 <= p1[0] < width
                                and 0 <= p1[1] < height
                                and 0 <= p2[0] < width
                                and 0 <= p2[1] < height
                            ):
                                cv2.line(
                                    frame,
                                    p1,
                                    p2,
                                    (255, 150, 0),
                                    2
                                )

                cv2.putText(
                    frame,
                    f"Detected poses: {pose_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                writer.write(frame)
                frame_index += 1

                if frame_index % 50 == 0:
                    print(
                        f"Processed {frame_index} frames | "
                        f"Poses: {pose_count}"
                    )

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start_time

    print("\n========== RESULTS ==========")
    print("Frames processed:", frame_index)
    print("Frames with poses:", detected_frames)
    print("Total pose detections:", total_poses)
    print("Processing time:", round(elapsed, 2), "seconds")
    print(
        "Processing FPS:",
        round(frame_index / elapsed, 2) if elapsed > 0 else 0
    )
    print("Output:", output_path)


if __name__ == "__main__":
    main()
