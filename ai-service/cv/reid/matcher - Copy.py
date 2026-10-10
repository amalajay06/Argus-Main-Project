import numpy as np


class ReIDMatcher:
    """
    Matches local BoT-SORT track embeddings against
    previously registered global identities.

    Threshold-based merging is intentionally disabled
    until proper validation data is available.
    """

    def __init__(self):
        self.identities = {}
        self.next_person_id = 1

    def normalize(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)

        if norm == 0:
            return embedding

        return embedding / norm

    def register_identity(self, embedding, class_name="person"):
        embedding = self.normalize(embedding)

        global_id = f"{class_name}_{self.next_person_id:04d}"

        self.identities[global_id] = {
            "class": class_name,
            "embedding": embedding
        }

        self.next_person_id += 1

        return global_id

    def compare(self, embedding, class_name="person"):
        embedding = self.normalize(embedding)

        results = []

        for global_id, identity in self.identities.items():

            if identity["class"] != class_name:
                continue

            similarity = float(
                np.dot(
                    embedding,
                    identity["embedding"]
                )
            )

            results.append({
                "global_id": global_id,
                "similarity": similarity
            })

        results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return results

    def get_best_match(self, embedding, class_name="person"):
        matches = self.compare(
            embedding,
            class_name
        )

        if not matches:
            return None

        return matches[0]