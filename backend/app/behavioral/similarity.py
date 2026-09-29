"""
Behavioral Fingerprint Similarity Module (Phase 6a).

Provides metric spaces and search routines for behavioral fingerprints:
  - Cosine Similarity (normalized to [0, 1])
  - Euclidean Distance (L2)
  - Manhattan Distance (L1)
  - O(N) Nearest Neighbor Similarity Search with Per-Dimension Z-Score Normalization
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from app.core.schemas import BehavioralFingerprint
from app.graph.networkx_store import NetworkXStore
from app.behavioral.fingerprint import extract_fingerprint, extract_fingerprint_safe


def cosine_similarity(fp1: Any, fp2: Any) -> float:
    """
    Computes cosine similarity between two behavioral fingerprints or vectors,
    scaled symmetrically into the interval [0.0, 1.0].
    """
    v1 = fp1.to_vector() if hasattr(fp1, "to_vector") else np.asarray(fp1, dtype=np.float32)
    v2 = fp2.to_vector() if hasattr(fp2, "to_vector") else np.asarray(fp2, dtype=np.float32)

    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0

    raw_cos = float(np.dot(v1, v2) / (norm1 * norm2))
    # Symmetrically map from [-1.0, +1.0] to [0.0, 1.0]
    sim = 0.5 * (raw_cos + 1.0)
    return float(np.clip(sim, 0.0, 1.0))


def euclidean_distance(fp1: Any, fp2: Any) -> float:
    """Computes Euclidean (L2) distance between two fingerprints."""
    v1 = fp1.to_vector() if hasattr(fp1, "to_vector") else np.asarray(fp1, dtype=np.float32)
    v2 = fp2.to_vector() if hasattr(fp2, "to_vector") else np.asarray(fp2, dtype=np.float32)
    return float(np.linalg.norm(v1 - v2))


def manhattan_distance(fp1: Any, fp2: Any) -> float:
    """Computes Manhattan (L1) distance between two fingerprints."""
    v1 = fp1.to_vector() if hasattr(fp1, "to_vector") else np.asarray(fp1, dtype=np.float32)
    v2 = fp2.to_vector() if hasattr(fp2, "to_vector") else np.asarray(fp2, dtype=np.float32)
    return float(np.sum(np.abs(v1 - v2)))


def find_similar_wallets(
    target_address: str,
    store: NetworkXStore,
    top_k: int = 5,
    min_txs: int = 3
) -> List[Tuple[str, float]]:
    """
    Scans all wallets in the store with >= min_txs transactions,
    z-score normalizes feature dimensions across the population,
    and returns top_k most similar wallets to target_address.
    """
    # 1. Obtain target fingerprint
    target_fp = extract_fingerprint_safe(store, target_address)
    if target_fp is None:
        return []
    target_vec = target_fp.to_vector()

    # 2. Extract population fingerprints (>= min_txs)
    addresses: List[str] = []
    vectors: List[np.ndarray] = []

    for addr in store.wallets.keys():
        if addr == target_address:
            continue
        try:
            fp = extract_fingerprint(store, addr, min_txs=min_txs)
            addresses.append(addr)
            vectors.append(fp.to_vector())
        except ValueError:
            continue

    if not vectors:
        return []

    X = np.array(vectors, dtype=np.float32)  # shape (N, 64)

    # 3. Per-dimension z-score normalization across population
    mean_vec = np.mean(X, axis=0)
    std_vec = np.std(X, axis=0)
    eps = 1e-6

    # Normalize population and target
    X_norm = (X - mean_vec) / (std_vec + eps)
    target_norm = (target_vec - mean_vec) / (std_vec + eps)

    # 4. Compute cosine similarity against all candidates
    target_norm_len = float(np.linalg.norm(target_norm))
    if target_norm_len < 1e-9:
        sims = np.zeros(len(addresses), dtype=np.float32)
    else:
        dots = np.dot(X_norm, target_norm)
        row_norms = np.linalg.norm(X_norm, axis=1)
        sims = 0.5 * (dots / (row_norms * target_norm_len + eps) + 1.0)
        sims = np.clip(sims, 0.0, 1.0)

    # 5. Rank and return top_k
    ranked_indices = np.argsort(sims)[::-1][:top_k]
    return [(addresses[i], float(sims[i])) for i in ranked_indices]
