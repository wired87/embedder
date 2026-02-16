import json

import numpy as np
from functools import lru_cache
EMBEDDER = None
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedder(dim=7):
    if EMBEDDER is None:
        if dim == 7:
            model="all-mpnet-base-v2"
        else:
            model="all-MiniLM-L6-v2"
        return SentenceTransformer(model)
    else:
        return EMBEDDER

EMBEDDER = get_embedder()

@lru_cache(maxsize=1)
def embed(text):
    if isinstance(text, dict):
        text=json.dumps(text)
    return np.array(EMBEDDER.encode(str(text.lower())), dtype=np.float64)


@lru_cache(maxsize=128)  # maxsize=1 ist oft zu wenig für Embeddings
def similarity(vec1_tuple, vec2_tuple):
    # Wir wandeln die Tupel erst hier in Arrays um für die Berechnung
    v1 = np.array(vec1_tuple)
    v2 = np.array(vec2_tuple)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


