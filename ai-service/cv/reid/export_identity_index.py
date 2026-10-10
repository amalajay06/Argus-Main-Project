import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from cv.reid.identity_registry import ReIDIdentityRegistry


OUTPUT_DIR = PROJECT_ROOT / "outputs"

REGISTRY_FILE = (
    OUTPUT_DIR / "reid_identity_registry.json"
)

INDEX_FILE = (
    OUTPUT_DIR / "reid_identity_index.json"
)

SOURCE_JSON = (
    OUTPUT_DIR / "test1_yolov8n.json"
)

ENRICHED_JSON = (
    OUTPUT_DIR / "test1_yolov8n_with_global_ids.json"
)


def export_identity_index():

    registry = ReIDIdentityRegistry(
        registry_path=REGISTRY_FILE
    )

    if not registry.load():
        raise RuntimeError(
            f"Could not load registry: "
            f"{REGISTRY_FILE}"
        )

    with open(
        REGISTRY_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        registry_data = json.load(file)

    identities = {}

    embedding_to_global = {}

    for identity in registry_data.get(
        "identities",
        []
    ):

        global_id = identity[
            "global_id"
        ]

        identities[global_id] = {
            "global_id": global_id,
            "class": identity.get(
                "class"
            ),
            "video_id": (
                identity.get("video_ids", [None])[0]
                if identity.get("video_ids")
                else None
            ),
            "track_ids": identity.get(
                "track_ids",
                []
            ),
            "embeddings": []
        }

        for embedding_id in identity.get(
            "embedding_ids",
            []
        ):

            embedding_to_global[
                embedding_id
            ] = global_id

            identities[global_id][
                "embeddings"
            ].append(
                {
                    "embedding_id": embedding_id
                }
            )

    output = {
        "version": 1,
        "automatic_merging": False,
        "total_identities": len(
            identities
        ),
        "identities": list(
            identities.values()
        )
    }

    with open(
        INDEX_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2
        )

    # ------------------------------------------------------------
    # ENRICH DETECTION JSON
    # ------------------------------------------------------------

    if SOURCE_JSON.exists():

        with open(
            SOURCE_JSON,
            "r",
            encoding="utf-8"
        ) as file:
            detection_data = json.load(
                file
            )

        assigned = 0

        for frame in detection_data.get(
            "frames",
            []
        ):

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

                global_id = (
                    embedding_to_global.get(
                        embedding_id
                    )
                )

                if global_id is None:
                    continue

                detection[
                    "global_id"
                ] = global_id

                assigned += 1

        detection_data[
            "identity_registry"
        ] = {
            "registry_file":
                str(
                    REGISTRY_FILE.name
                ),
            "automatic_merging":
                False
        }

        with open(
            ENRICHED_JSON,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                detection_data,
                file,
                indent=2
            )

        print(
            f"Enriched JSON saved: "
            f"{ENRICHED_JSON}"
        )

        print(
            f"Global IDs assigned: "
            f"{assigned}"
        )

    print(
        f"Identity index saved: "
        f"{INDEX_FILE}"
    )

    print(
        f"Total identities: "
        f"{len(identities)}"
    )


if __name__ == "__main__":
    export_identity_index()