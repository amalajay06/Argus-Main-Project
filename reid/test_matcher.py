import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.reid.matcher import ReIDMatcher


matcher = ReIDMatcher()

# Simulated normalized embeddings
embedding_a = np.random.rand(2048).astype(np.float32)
embedding_a /= np.linalg.norm(embedding_a)

embedding_b = embedding_a + (
    np.random.rand(2048).astype(np.float32) * 0.01
)

embedding_b /= np.linalg.norm(embedding_b)


# Register first identity
global_id = matcher.register_identity(
    embedding_a,
    "person"
)

print("Registered:", global_id)


# Search for matching identity
matches = matcher.compare(
    embedding_b,
    "person"
)

print("\nMatches:")

for match in matches:
    print(
        match["global_id"],
        f"{match['similarity']:.4f}"
    )


print("\nBest match:")

best = matcher.get_best_match(
    embedding_b,
    "person"
)

print(best)