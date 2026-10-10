import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from cv.reid.identity_registry import ReIDIdentityRegistry


OUTPUT_DIR = PROJECT_ROOT / "outputs"

JSON_FILE = (
    OUTPUT_DIR / "test1_yolov8n.json"
)

EMBEDDINGS_DIR = (
    OUTPUT_DIR / "reid_embeddings"
)

REGISTRY_FILE = (
    OUTPUT_DIR / "reid_identity_registry.json"
)


def build_registry():

    if not JSON_FILE.exists():
        raise FileNotFoundError(
            f"JSON not found: {JSON_FILE}"
        )

    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(
            f"Embedding directory not found: "
            f"{EMBEDDINGS_DIR}"
        )

    with open(
        JSON_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    registry = ReIDIdentityRegistry(
        registry_path=REGISTRY_FILE
    )

    # ------------------------------------------------------------
    # COLLECT EMBEDDINGS BY VIDEO + CLASS + TRACK
    # ------------------------------------------------------------

    tracks = {}

    for frame in data.get("frames", []):

        frame_idx = frame.get(
            "frame_idx"
        )

        for detection in frame.get(
            "detections",
            []
        ):

            reid = detection.get(
                "reid",
                {}
            )

            embedding_id = reid.get(
                "embedding_id"
            )

            if not embedding_id:
                continue

            track_id = detection.get(
                "track_id"
            )

            if track_id is None:
                continue

            class_name = detection.get(
                "class"
            )

            if class_name == "person":
                reid_class = "person"

            elif class_name in {
                "car",
                "truck",
                "bus",
                "motorcycle"
            }:
                reid_class = "vehicle"

            else:
                continue

            embedding_path = (
                EMBEDDINGS_DIR
                / f"{embedding_id}.npy"
            )

            if not embedding_path.exists():
                continue

            video_id = data.get(
                "video_id"
            )

            key = (
                video_id,
                reid_class,
                track_id
            )

            if key not in tracks:
                tracks[key] = []

            tracks[key].append({
                "embedding_id": embedding_id,
                "embedding_path": embedding_path,
                "frame_idx": frame_idx
            })

    # ------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------

    total_embeddings = sum(
        len(items)
        for items in tracks.values()
    )

    person_tracks = sum(
        1
        for key in tracks
        if key[1] == "person"
    )

    vehicle_tracks = sum(
        1
        for key in tracks
        if key[1] == "vehicle"
    )

    print(
        f"Tracks found: {len(tracks)}"
    )

    print(
        f"Person tracks: {person_tracks}"
    )

    print(
        f"Vehicle tracks: {vehicle_tracks}"
    )

    print(
        f"Embeddings: {total_embeddings}"
    )

    # ------------------------------------------------------------
    # CREATE REGISTRY
    # ------------------------------------------------------------

    registry.matcher = (
        __import__(
            "cv.reid.matcher",
            fromlist=["ReIDMatcher"]
        ).ReIDMatcher()
    )

    registry.embedding_metadata = {}

    # ------------------------------------------------------------
    # ONE GLOBAL ID PER LOCAL TRACK
    # ------------------------------------------------------------

    for (
        video_id,
        class_name,
        track_id
    ), embeddings in sorted(
        tracks.items(),
        key=lambda item: (
            item[0][1],
            item[0][2]
        )
    ):

        embeddings.sort(
            key=lambda item: item["frame_idx"]
        )

        first = embeddings[0]

        first_embedding = np.load(
            first["embedding_path"]
        )

        global_id = (
            registry.register_identity(
                embedding=first_embedding,
                class_name=class_name,
                embedding_id=first[
                    "embedding_id"
                ],
                track_id=track_id,
                video_id=video_id
            )
        )

        for item in embeddings[1:]:

            embedding = np.load(
                item["embedding_path"]
            )

            registry.add_embedding(
                global_id=global_id,
                embedding=embedding,
                embedding_id=item[
                    "embedding_id"
                ],
                track_id=track_id,
                video_id=video_id
            )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    registry.save()

    print(
        f"Registry saved: {REGISTRY_FILE}"
    )

    print(
        registry.summary()
    )


if __name__ == "__main__":
    build_registry()