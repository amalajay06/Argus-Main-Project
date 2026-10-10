import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------
# Make ai-service available to Python
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.reid.extractor import ReIDExtractor


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

VIDEO_PATH = PROJECT_ROOT / "data" / "test1.mp4"

OUTPUT_DIR = PROJECT_ROOT / "reid" / "track_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

MODEL_NAME = "yolov8n.pt"

CONFIDENCE = 0.60

MIN_WIDTH = 80
MIN_HEIGHT = 150

# Test only a small number of tracks.
MAX_TRACKS = 5

# Maximum number of Re-ID observations per track.
MAX_EMBEDDINGS_PER_TRACK = 5

# Minimum number of frames between Re-ID samples
# for the same track.
MIN_FRAME_GAP = 15


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

print("Loading YOLO...")

yolo = YOLO(MODEL_NAME)

print("Loading FastReID...")

reid = ReIDExtractor()

print("Models ready.")


# ------------------------------------------------------------
# Track storage
#
# track_id -> {
#     "embeddings": [],
#     "frames": [],
#     "confidences": [],
#     "boxes": []
# }
# ------------------------------------------------------------

track_data = {}

frame_idx = 0


# ------------------------------------------------------------
# Process video with BoT-SORT
# ------------------------------------------------------------

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

    for box, class_id, confidence, track_id in zip(
        boxes,
        classes,
        confidences,
        track_ids
    ):

        # --------------------------------------------------------
        # Person only
        # --------------------------------------------------------

        if int(class_id) != 0:
            continue

        confidence = float(confidence)

        if confidence < CONFIDENCE:
            continue

        # --------------------------------------------------------
        # Stop discovering new tracks after MAX_TRACKS
        # --------------------------------------------------------

        if (
            track_id not in track_data
            and len(track_data) >= MAX_TRACKS
        ):
            continue

        # --------------------------------------------------------
        # Initialize track
        # --------------------------------------------------------

        if track_id not in track_data:

            track_data[track_id] = {
                "embeddings": [],
                "frames": [],
                "confidences": [],
                "boxes": []
            }

        data = track_data[track_id]

        # --------------------------------------------------------
        # Maximum samples reached
        # --------------------------------------------------------

        if len(data["embeddings"]) >= MAX_EMBEDDINGS_PER_TRACK:
            continue

        # --------------------------------------------------------
        # Temporal spacing
        # --------------------------------------------------------

        if data["frames"]:

            last_frame = data["frames"][-1]

            if frame_idx - last_frame < MIN_FRAME_GAP:
                continue

        # --------------------------------------------------------
        # Bounding box
        # --------------------------------------------------------

        x1, y1, x2, y2 = map(int, box)

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop_width = x2 - x1
        crop_height = y2 - y1

        # Ignore small crops
        if crop_width < MIN_WIDTH or crop_height < MIN_HEIGHT:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # --------------------------------------------------------
        # Save crop
        # --------------------------------------------------------

        crop_path = (
            OUTPUT_DIR
            / f"track_{track_id}_frame_{frame_idx}.jpg"
        )

        cv2.imwrite(str(crop_path), crop)

        # --------------------------------------------------------
        # Extract Re-ID embedding
        # --------------------------------------------------------

        embedding = reid.extract(crop_path)

        data["embeddings"].append(
            embedding.numpy()
        )

        data["frames"].append(
            frame_idx
        )

        data["confidences"].append(
            confidence
        )

        data["boxes"].append(
            [x1, y1, x2, y2]
        )

        print(
            f"Track {track_id}: "
            f"embedding "
            f"{len(data['embeddings'])}/"
            f"{MAX_EMBEDDINGS_PER_TRACK} "
            f"at frame {frame_idx} "
            f"(conf={confidence:.3f})"
        )

    frame_idx += 1


# ------------------------------------------------------------
# Build representative embedding per track
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRACK-LEVEL RE-ID RESULTS")
print("=" * 60)


track_embeddings = {}


for track_id, data in track_data.items():

    embeddings = data["embeddings"]

    if not embeddings:
        continue

    matrix = np.stack(embeddings)

    # --------------------------------------------------------
    # Average embeddings from multiple observations
    # --------------------------------------------------------

    representative = np.mean(matrix, axis=0)

    # L2 normalize
    norm = np.linalg.norm(representative)

    if norm > 0:
        representative = representative / norm

    track_embeddings[track_id] = representative

    print(
        f"\nTrack ID: {track_id}"
    )

    print(
        f"Observations: {len(embeddings)}"
    )

    print(
        f"Frames: {data['frames']}"
    )

    print(
        f"Confidences: "
        f"{[round(x, 3) for x in data['confidences']]}"
    )

    print(
        f"Embedding shape: {representative.shape}"
    )

    print(
        f"Embedding norm: "
        f"{np.linalg.norm(representative):.6f}"
    )

    print(
        "First 5 values:",
        representative[:5]
    )


# ------------------------------------------------------------
# Compare track embeddings
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRACK SIMILARITY")
print("=" * 60)


track_ids = list(track_embeddings.keys())


for i in range(len(track_ids)):

    for j in range(i + 1, len(track_ids)):

        id_a = track_ids[i]
        id_b = track_ids[j]

        emb_a = track_embeddings[id_a]
        emb_b = track_embeddings[id_b]

        similarity = float(
            np.dot(emb_a, emb_b)
        )

        print(
            f"Track {id_a} vs Track {id_b}: "
            f"cosine similarity = {similarity:.4f}"
        )


print("\nTrack-level Re-ID test completed.")