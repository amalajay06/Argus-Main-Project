import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------
# Project path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.reid.extractor import ReIDExtractor


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

VIDEO_PATH = PROJECT_ROOT / "data" / "test1.mp4"

MODEL_NAME = "yolov8n.pt"

CONFIDENCE = 0.60

MIN_WIDTH = 80
MIN_HEIGHT = 150

# Number of BoT-SORT tracks we want to study
MAX_TRACKS = 20

# Number of Re-ID observations collected per track
OBSERVATIONS_PER_TRACK = 5

# Minimum number of video frames between observations
MIN_FRAME_GAP = 30


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
# ------------------------------------------------------------

track_data = {}

frame_idx = 0


# ------------------------------------------------------------
# Process entire video
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
        # Don't create unlimited tracks
        # --------------------------------------------------------

        if (
            track_id not in track_data
            and len(track_data) >= MAX_TRACKS
        ):
            continue

        if track_id not in track_data:

            track_data[track_id] = {
                "embeddings": [],
                "frames": [],
                "confidences": []
            }

        data = track_data[track_id]

        # --------------------------------------------------------
        # Already enough observations?
        # --------------------------------------------------------

        if len(data["embeddings"]) >= OBSERVATIONS_PER_TRACK:
            continue

        # --------------------------------------------------------
        # Keep observations temporally separated
        # --------------------------------------------------------

        if data["frames"]:

            if frame_idx - data["frames"][-1] < MIN_FRAME_GAP:
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

        if crop_width < MIN_WIDTH:
            continue

        if crop_height < MIN_HEIGHT:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # --------------------------------------------------------
        # Save crop
        # --------------------------------------------------------

        crop_path = (
            PROJECT_ROOT
            / "reid"
            / "full_video_crops"
            / f"track_{track_id}_frame_{frame_idx}.jpg"
        )

        crop_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

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

        print(
            f"Track {track_id}: "
            f"{len(data['embeddings'])}/"
            f"{OBSERVATIONS_PER_TRACK} "
            f"at frame {frame_idx}"
        )

    frame_idx += 1


# ------------------------------------------------------------
# Select valid tracks
# ------------------------------------------------------------

valid_tracks = {
    track_id: data
    for track_id, data in track_data.items()
    if len(data["embeddings"]) >= 2
}


print("\n" + "=" * 70)
print("COLLECTED TRACKS")
print("=" * 70)

for track_id, data in valid_tracks.items():

    print(
        f"Track {track_id}: "
        f"{len(data['embeddings'])} observations | "
        f"frames={data['frames']}"
    )


# ------------------------------------------------------------
# Same-track similarities
# ------------------------------------------------------------

same_track_scores = []


for track_id, data in valid_tracks.items():

    embeddings = data["embeddings"]

    for i in range(len(embeddings)):

        for j in range(i + 1, len(embeddings)):

            similarity = float(
                np.dot(
                    embeddings[i],
                    embeddings[j]
                )
            )

            same_track_scores.append(
                similarity
            )


# ------------------------------------------------------------
# Different-track similarities
# ------------------------------------------------------------

different_track_scores = []

track_ids = list(valid_tracks.keys())


for i in range(len(track_ids)):

    for j in range(i + 1, len(track_ids)):

        track_a = valid_tracks[track_ids[i]]
        track_b = valid_tracks[track_ids[j]]

        for emb_a in track_a["embeddings"]:

            for emb_b in track_b["embeddings"]:

                similarity = float(
                    np.dot(
                        emb_a,
                        emb_b
                    )
                )

                different_track_scores.append(
                    similarity
                )


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

def print_statistics(name, scores):

    if not scores:

        print(f"\n{name}: no scores available")

        return

    scores = np.array(scores)

    print("\n" + name)

    print("-" * 50)

    print("Count:", len(scores))

    print(
        "Minimum:",
        f"{scores.min():.4f}"
    )

    print(
        "Maximum:",
        f"{scores.max():.4f}"
    )

    print(
        "Mean:",
        f"{scores.mean():.4f}"
    )

    print(
        "Median:",
        f"{np.median(scores):.4f}"
    )

    print(
        "25th percentile:",
        f"{np.percentile(scores, 25):.4f}"
    )

    print(
        "75th percentile:",
        f"{np.percentile(scores, 75):.4f}"
    )


# ------------------------------------------------------------
# Similarity results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RE-ID SIMILARITY ANALYSIS")
print("=" * 70)

print_statistics(
    "SAME-TRACK SIMILARITY",
    same_track_scores
)

print_statistics(
    "DIFFERENT-TRACK SIMILARITY",
    different_track_scores
)


# ------------------------------------------------------------
# Distribution overlap
# ------------------------------------------------------------

if same_track_scores and different_track_scores:

    same_min = min(same_track_scores)
    same_max = max(same_track_scores)

    different_min = min(different_track_scores)
    different_max = max(different_track_scores)

    print("\n" + "=" * 70)
    print("IDENTITY SEPARATION")
    print("=" * 70)

    print(
        f"Same-track range:      "
        f"{same_min:.4f} → {same_max:.4f}"
    )

    print(
        f"Different-track range: "
        f"{different_min:.4f} → {different_max:.4f}"
    )

    if same_min > different_max:

        print(
            "\nNo overlap was observed "
            "in this sample."
        )

    else:

        print(
            "\nSimilarity distributions overlap."
        )

        print(
            "A fixed threshold should NOT be "
            "selected from this experiment alone."
        )


print("\nRe-ID similarity experiment completed.")