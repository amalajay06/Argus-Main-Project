import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cv.reid.extractor import ReIDExtractor

VIDEO_PATH = PROJECT_ROOT / "data" / "test1.mp4"
MODEL_NAME = "yolov8n.pt"

CONFIDENCE = 0.60
MIN_WIDTH = 80
MIN_HEIGHT = 150
MIN_FRAME_GAP = 30

MAX_IDENTITIES = 10
OBSERVATIONS_PER_IDENTITY = 3


print("Loading YOLO...")
yolo = YOLO(MODEL_NAME)

print("Loading FastReID...")
reid = ReIDExtractor()

print("Models ready.")

identities = {}
frame_idx = 0

results = yolo.track(
    source=str(VIDEO_PATH),
    tracker="botsort.yaml",
    stream=True,
    persist=True,
    device=0,
    verbose=False
)

for result in results:

    frame = result.orig_img

    if result.boxes is None or result.boxes.id is None:
        frame_idx += 1
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)

    for box, cls, conf, track_id in zip(
        boxes, classes, confidences, track_ids
    ):

        if int(cls) != 0:
            continue

        conf = float(conf)

        if conf < CONFIDENCE:
            continue

        if track_id not in identities:

            if len(identities) >= MAX_IDENTITIES:
                continue

            identities[track_id] = {
                "embeddings": [],
                "frames": []
            }

        data = identities[track_id]

        if len(data["embeddings"]) >= OBSERVATIONS_PER_IDENTITY:
            continue

        if data["frames"]:
            if frame_idx - data["frames"][-1] < MIN_FRAME_GAP:
                continue

        x1, y1, x2, y2 = map(int, box)

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
            PROJECT_ROOT
            / "reid"
            / "matcher_crops"
            / f"track_{track_id}_{frame_idx}.jpg"
        )

        crop_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(str(crop_path), crop)

        embedding = reid.extract(crop_path).numpy()

        data["embeddings"].append(embedding)
        data["frames"].append(frame_idx)

        print(
            f"Track {track_id}: "
            f"{len(data['embeddings'])}/{OBSERVATIONS_PER_IDENTITY}"
        )

    frame_idx += 1


# ------------------------------------------------------------
# Create one representative embedding per track
# ------------------------------------------------------------

representatives = {}

for track_id, data in identities.items():

    if len(data["embeddings"]) == 0:
        continue

    embedding = np.mean(
        np.stack(data["embeddings"]),
        axis=0
    )

    embedding = embedding / np.linalg.norm(embedding)

    representatives[track_id] = embedding


# ------------------------------------------------------------
# Match every track against every other track
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RE-ID MATCHING RESULTS")
print("=" * 70)

for track_id, embedding in representatives.items():

    scores = []

    for other_id, other_embedding in representatives.items():

        if track_id == other_id:
            continue

        similarity = float(
            np.dot(embedding, other_embedding)
        )

        scores.append(
            (other_id, similarity)
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nTrack {track_id}")
    print("-" * 40)

    for rank, (other_id, similarity) in enumerate(
        scores[:5],
        start=1
    ):

        print(
            f"{rank}. Track {other_id} "
            f"similarity={similarity:.4f}"
        )


print("\nMatching experiment completed.")