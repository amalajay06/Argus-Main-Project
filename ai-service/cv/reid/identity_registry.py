import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from cv.reid.matcher import ReIDMatcher


class ReIDIdentityRegistry:
    """
    Persistent registry for Argus Re-ID identities.

    Stores global identity metadata in JSON and loads the
    corresponding embedding vectors from .npy files.

    Automatic identity merging is intentionally disabled.
    """

    def __init__(
        self,
        registry_path="outputs/reid_identity_registry.json",
        embeddings_dir="outputs/reid_embeddings"
    ):
        self.registry_path = (
            PROJECT_ROOT / registry_path
        )

        self.embeddings_dir = (
            PROJECT_ROOT / embeddings_dir
        )

        self.matcher = ReIDMatcher()

        self.embedding_metadata = {}

    # ============================================================
    # REGISTER IDENTITY
    # ============================================================

    def register_identity(
        self,
        embedding,
        class_name,
        embedding_id=None,
        track_id=None,
        video_id=None
    ):
        global_id = self.matcher.register_identity(
            embedding=embedding,
            class_name=class_name,
            track_id=track_id,
            video_id=video_id,
            embedding_id=embedding_id
        )

        if embedding_id is not None:
            self.embedding_metadata[
                embedding_id
            ] = {
                "global_id": global_id,
                "class": class_name,
                "track_id": track_id,
                "video_id": video_id
            }

        return global_id

    # ============================================================
    # ADD EMBEDDING
    # ============================================================

    def add_embedding(
        self,
        global_id,
        embedding,
        embedding_id=None,
        track_id=None,
        video_id=None
    ):
        self.matcher.add_embedding(
            global_id=global_id,
            embedding=embedding,
            track_id=track_id,
            video_id=video_id,
            embedding_id=embedding_id
        )

        if embedding_id is not None:
            identity = self.matcher.get_identity(
                global_id
            )

            self.embedding_metadata[
                embedding_id
            ] = {
                "global_id": global_id,
                "class": identity["class"],
                "track_id": track_id,
                "video_id": video_id
            }

    # ============================================================
    # COMPARE
    # ============================================================

    def compare(
        self,
        embedding,
        class_name
    ):
        return self.matcher.compare(
            embedding,
            class_name
        )

    def get_best_match(
        self,
        embedding,
        class_name
    ):
        return self.matcher.get_best_match(
            embedding,
            class_name
        )

    # ============================================================
    # SAVE
    # ============================================================

    def save(self):
        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "version": 1,
            "automatic_merging": False,
            "identities": (
                self.matcher.get_all_identities()
            ),
            "embeddings": self.embedding_metadata
        }

        with open(
            self.registry_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=2
            )

    # ============================================================
    # LOAD
    # ============================================================

    def load(self):
        if not self.registry_path.exists():
            return False

        with open(
            self.registry_path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        self.matcher = ReIDMatcher()

        self.embedding_metadata = data.get(
            "embeddings",
            {}
        )

        identities = data.get(
            "identities",
            []
        )

        for identity in identities:

            global_id = identity["global_id"]
            class_name = identity["class"]

            embedding_ids = identity.get(
                "embedding_ids",
                []
            )

            track_ids = identity.get(
                "track_ids",
                []
            )

            video_ids = identity.get(
                "video_ids",
                []
            )

            embeddings = []

            for embedding_id in embedding_ids:

                embedding_path = (
                    self.embeddings_dir
                    / f"{embedding_id}.npy"
                )

                if not embedding_path.exists():
                    continue

                embedding = np.load(
                    embedding_path
                )

                embeddings.append(
                    self.matcher.normalize(
                        embedding
                    )
                )

            if not embeddings:
                continue

            self.matcher.identities[
                global_id
            ] = {
                "class": class_name,
                "embeddings": embeddings,
                "track_ids": track_ids,
                "video_ids": video_ids,
                "embedding_ids": embedding_ids
            }

            # Keep future generated IDs unique.
            try:
                number = int(
                    global_id.rsplit(
                        "_",
                        1
                    )[1]
                )

                current = self.matcher.next_ids.get(
                    class_name,
                    1
                )

                self.matcher.next_ids[
                    class_name
                ] = max(
                    current,
                    number + 1
                )

            except (ValueError, IndexError):
                pass

        return True

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(self):
        return {
            "total_identities":
                self.matcher.count_identities(),

            "person_identities":
                self.matcher.count_identities(
                    "person"
                ),

            "vehicle_identities":
                self.matcher.count_identities(
                    "vehicle"
                ),

            "embeddings":
                len(self.embedding_metadata)
        }


if __name__ == "__main__":
    registry = ReIDIdentityRegistry()

    loaded = registry.load()

    print(
        "Loaded:",
        loaded
    )

    print(
        registry.summary()
    )