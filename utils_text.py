import re
import numpy as np

def normalize_kw(k: str) -> str:
    k = k.strip()
    k = re.sub(r"\s+", " ", k)
    return k

def dedupe_keywords(keywords, embeddings, threshold=0.9):
    kept = []
    keep_mask = np.ones(len(keywords), dtype=bool)
    for i in range(len(keywords)):
        if not keep_mask[i]:
            continue
        for j in kept:
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                keep_mask[i] = False
                break
        if keep_mask[i]:
            kept.append(i)
    return keep_mask
