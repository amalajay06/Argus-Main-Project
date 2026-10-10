import numpy as np


class ReIDMatcher:
    """
    Maintains global Re-ID identities for Argus.

    Responsibilities:
    - Normalize Re-ID embeddings.
    - Register new global identities.
    - Compare an embedding against existing identities.
    - Maintain multiple embeddings for the same identity.
    - Keep person and vehicle identities separate.
    - Return similarity information for downstream validation.

    Automatic threshold-based merging is intentionally disabled.
    A validated threshold should be introduced only after proper
    person/vehicle Re-ID evaluation.
    """

    def __init__(self):
        self.identities = {}

        self.next_ids = {
            "person": 1,
            "vehicle": 1
        }

    # ============================================================
    # EMBEDDING UTILITIES
    # ============================================================

    def normalize(self, embedding):
        """
        L2-normalize an embedding.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return embedding

        return embedding / norm

    # ============================================================
    # GLOBAL ID CREATION
    # ============================================================

    def _create_global_id(self, class_name):
        """
        Create a class-specific global identity ID.
        """

        if class_name not in self.next_ids:
            self.next_ids[class_name] = 1

        global_id = (
            f"{class_name}_"
            f"{self.next_ids[class_name]:04d}"
        )

        self.next_ids[class_name] += 1

        return global_id

    def register_identity(
        self,
        embedding,
        class_name="person",
        track_id=None,
        video_id=None,
        embedding_id=None
    ):
        """
        Register a new global identity.

        This function does NOT determine whether the identity
        already exists. The caller must make that decision.
        """

        embedding = self.normalize(embedding)

        global_id = self._create_global_id(
            class_name
        )

        self.identities[global_id] = {
            "class": class_name,
            "embeddings": [
                embedding
            ],
            "track_ids": (
                [track_id]
                if track_id is not None
                else []
            ),
            "video_ids": (
                [video_id]
                if video_id is not None
                else []
            ),
            "embedding_ids": (
                [embedding_id]
                if embedding_id is not None
                else []
            )
        }

        return global_id

    # ============================================================
    # ADD EMBEDDING TO EXISTING IDENTITY
    # ============================================================

    def add_embedding(
        self,
        global_id,
        embedding,
        track_id=None,
        video_id=None,
        embedding_id=None
    ):
        """
        Add another Re-ID embedding to an existing identity.
        """

        if global_id not in self.identities:
            raise KeyError(
                f"Unknown global identity: {global_id}"
            )

        identity = self.identities[global_id]

        embedding = self.normalize(embedding)

        identity["embeddings"].append(
            embedding
        )

        if (
            track_id is not None
            and track_id not in identity["track_ids"]
        ):
            identity["track_ids"].append(
                track_id
            )

        if (
            video_id is not None
            and video_id not in identity["video_ids"]
        ):
            identity["video_ids"].append(
                video_id
            )

        if (
            embedding_id is not None
            and embedding_id not in identity["embedding_ids"]
        ):
            identity["embedding_ids"].append(
                embedding_id
            )

    # ============================================================
    # COMPARE AGAINST GLOBAL IDENTITIES
    # ============================================================

    def compare(
        self,
        embedding,
        class_name="person"
    ):
        """
        Compare an embedding against all identities
        belonging to the requested class.

        For identities with multiple embeddings, the highest
        similarity is returned.
        """

        embedding = self.normalize(
            embedding
        )

        results = []

        for global_id, identity in self.identities.items():

            if identity["class"] != class_name:
                continue

            similarities = []

            for stored_embedding in identity[
                "embeddings"
            ]:
                similarity = float(
                    np.dot(
                        embedding,
                        stored_embedding
                    )
                )

                similarities.append(
                    similarity
                )

            if not similarities:
                continue

            best_similarity = max(
                similarities
            )

            results.append({
                "global_id": global_id,
                "class": class_name,
                "similarity": best_similarity,
                "embedding_count": len(
                    identity["embeddings"]
                )
            })

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return results

    # ============================================================
    # BEST MATCH
    # ============================================================

    def get_best_match(
        self,
        embedding,
        class_name="person"
    ):
        """
        Return the highest-similarity identity.

        This is ONLY a similarity result.
        It does not automatically assign the global ID.
        """

        matches = self.compare(
            embedding,
            class_name
        )

        if not matches:
            return None

        return matches[0]

    # ============================================================
    # IDENTITY INFORMATION
    # ============================================================

    def get_identity(self, global_id):
        """
        Return metadata for a global identity.
        """

        return self.identities.get(
            global_id
        )

    def get_all_identities(
        self,
        class_name=None
    ):
        """
        Return a lightweight summary of all
        registered identities.

        Embedding vectors themselves are excluded.
        """

        identities = []

        for global_id, identity in self.identities.items():

            if (
                class_name is not None
                and identity["class"] != class_name
            ):
                continue

            identities.append({
                "global_id": global_id,
                "class": identity["class"],
                "embedding_count": len(
                    identity["embeddings"]
                ),
                "track_ids": list(
                    identity["track_ids"]
                ),
                "video_ids": list(
                    identity["video_ids"]
                ),
                "embedding_ids": list(
                    identity["embedding_ids"]
                )
            })

        return identities

    # ============================================================
    # COUNTS
    # ============================================================

    def count_identities(
        self,
        class_name=None
    ):
        """
        Return the number of registered global identities.
        """

        if class_name is None:
            return len(
                self.identities
            )

        return sum(
            1
            for identity in self.identities.values()
            if identity["class"] == class_name
        )


if __name__ == "__main__":

    print(
        "Argus Re-ID matcher module loaded successfully."
    )