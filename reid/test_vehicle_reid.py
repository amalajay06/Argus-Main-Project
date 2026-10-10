import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.reid.vehicle_extractor import VehicleReIDExtractor


VIDEO_PATH = PROJECT_ROOT / "data" / "test1.mp4"
CROP_DIR = PROJECT_ROOT / "reid" / "vehicle_test_crops"

MODEL_NAME = "yolov8n.pt"

CONFIDENCE = 0.60
MIN_WIDTH = 100
MIN_HEIGHT = 60


print("Loading YOLO...")
yolo = YOLO(MODEL_NAME)

print("Loading Vehicle Re-ID...")
reid = VehicleReIDExtractor()

print("Models ready.")

CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results = yolo.track(
    source=str(VIDEO_PATH),
    tracker="botsort.yaml",
    stream=True,
    persist=True,
    device=0,
    verbose=False
)

found = False

for frame_idx, result in enumerate(results):

    if result.boxes is None or result.boxes.id is None:
        continue

    frame = result.orig_img

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)

    for box, class_id, confidence, track_id in zip(
        boxes,
        classes,
        confidences,
        track_ids
    ):

        class_name = yolo.names[int(class_id)]

        if class_name not in {
            "car",
            "truck",
            "bus",
            "motorcycle"
        }:
            continue

        if float(confidence) < CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(
            int,
            box
        )

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 - x1 < MIN_WIDTH:
            continue

        if y2 - y1 < MIN_HEIGHT:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_path = (
            CROP_DIR
            / f"vehicle_track_{track_id}_frame_{frame_idx}.jpg"
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        embedding = reid.extract_from_array(
            crop
        )

        print("\nVehicle Re-ID successful")

        print(
            "Class:",
            class_name
        )

        print(
            "Track ID:",
            int(track_id)
        )

        print(
            "Frame:",
            frame_idx
        )

        print(
            "Confidence:",
            round(float(confidence), 4)
        )

        print(
            "Embedding shape:",
            tuple(embedding.shape)
        )

        print(
            "Embedding dtype:",
            embedding.dtype
        )

        print(
            "Embedding norm:",
            float(embedding.norm())
        )

        print(
            "First 10 values:",
            embedding[:10].numpy()
        )

        print(
            "Crop:",
            crop_path
        )

        found = True
        break

    if found:
        break


if not found:
    print(
        "No suitable vehicle detection found."
    )