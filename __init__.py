import json

import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedder(dim=7):
    if dim == 7:
        model="all-mpnet-qf_core_base-v2"
    else:
        model="all-MiniLM-L6-v2"
    return SentenceTransformer(model)

EMBEDDER = get_embedder()

@lru_cache(maxsize=1)
def embed(text):
    if isinstance(text, dict):
        text=json.dumps(text)
    return np.array(EMBEDDER.encode(str(text.lower())), dtype=np.float64).tolist()

@lru_cache(maxsize=1)
def similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

