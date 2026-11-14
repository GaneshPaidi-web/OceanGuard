
from ultralytics import YOLO
import os
import time

# Load YOLO11 Nano
model = YOLO("yolo11n.pt")

# Input video
video_path = "test_images/person_test.mp4"

# Check video exists
if not os.path.exists(video_path):
    print("Video not found:", video_path)
    exit()

# Output directory
os.makedirs("outputs", exist_ok=True)

print("Starting YOLO video detection...")

start_time = time.time()

results = model.predict(
    source=video_path,
    imgsz=640,
    conf=0.05,
    iou=0.45,
    classes=[0],       # Person only
    device="cpu",
    save=True,
    project="outputs",
    name="yolo_video",
    exist_ok=True,
    stream=True
)

frame_count = 0
total_people = 0

for result in results:
    frame_count += 1
    people = len(result.boxes)
    total_people += people

    if frame_count % 30 == 0:
        print(
            f"Processed {frame_count} frames | "
            f"People in current frame: {people}"
        )

end_time = time.time()

print("\nVideo processing completed!")
print("Total frames:", frame_count)
print("Total person detections across frames:", total_people)
print("Processing time:", round(end_time - start_time, 2), "seconds")
print("Output folder: outputs/yolo_video/")
