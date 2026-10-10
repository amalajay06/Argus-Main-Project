import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO

from cv.reid.extractor import ReIDExtractor


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

VIDEO_PATH = Path("data/test1.mp4")
OUTPUT_DIR = Path("reid/test_crops")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "person_001.jpg"


# ------------------------------------------------------------
# YOLO
# ------------------------------------------------------------

model = YOLO("yolov8n.pt")


# ------------------------------------------------------------
# Open video
# ------------------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


found = False
frame_idx = 0


# ------------------------------------------------------------
# Find a reasonably large person
# ------------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        device=0,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:
        frame_idx += 1
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # COCO class 0 = person
        if class_id != 0:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        width = x2 - x1
        height = y2 - y1

        # Ignore tiny detections
        if width < 80 or height < 150:
            continue

        # Clamp coordinates
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        cv2.imwrite(
            str(OUTPUT_IMAGE),
            crop
        )

        print("Person crop saved:")
        print(f"  Frame: {frame_idx}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Bounding box: [{x1}, {y1}, {x2}, {y2}]")
        print(f"  File: {OUTPUT_IMAGE}")

        found = True
        break

    if found:
        break

    frame_idx += 1


cap.release()


if not found:
    raise RuntimeError(
        "No sufficiently large person detection was found."
    )


# ------------------------------------------------------------
# Re-ID
# ------------------------------------------------------------

print("\nLoading Re-ID extractor...")

extractor = ReIDExtractor()

print("Extracting embedding...")

embedding = extractor.extract(OUTPUT_IMAGE)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\nRe-ID inference successful!")
print("Embedding shape:", tuple(embedding.shape))
print("Embedding dtype:", embedding.dtype)
print("Embedding norm:", float(embedding.norm()))

print("\nFirst 10 embedding values:")
print(embedding[:10])