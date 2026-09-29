"""
Behavioral Fingerprint Extractor (Phase 6a).

Extracts a comprehensive 64-dimensional behavioral feature vector capturing
human habitual signatures across 4 key operational domains:
  1. Timing Vector (16 dims): inter-hop delays, interval histogram, burst speed
  2. Gas Fee Vector (16 dims): gas fee preference, quantization, outlier tolerance
  3. Amount Vector (16 dims): structuring patterns, round numbers, split ratios
  4. Chain & Temporal Vector (16 dims): chain preferences, active hours, diurnal cycles

Strict Architectural Invariant:
  This signal is strictly independent of topological proximity (proximity_rank)
  and tabular/graph ML models. It is NEVER blended.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import scipy.stats as stats

from app.core.schemas import BehavioralFingerprint, Transaction, Chain
from app.graph.networkx_store import NetworkXStore


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def extract_timing_vector(txs: List[Transaction]) -> Tuple[List[float], Dict[str, Any]]:
    """
    Extracts 16-dimensional timing vector from chronologically ordered transactions.
    - 10-bin log-spaced interval histogram
    - median, std, min, max, mean (5 dims)
    - burst_ratio: fraction of txs occurring within 60s of previous tx (1 dim)
    """
    timestamps = sorted([tx.timestamp for tx in txs])
    if len(timestamps) < 2:
        vec = [0.0] * 16
        summary = {"median_delay": 0.0, "burst_ratio": 0.0, "mean_delay": 0.0}
        return vec, summary

    intervals = np.diff(timestamps).astype(np.float64)
    # Filter negative intervals if any out-of-order timestamps occur
    intervals = np.maximum(intervals, 0.0)

    # 10 log-spaced bins from 1 second to ~100 days (~8.64e6 seconds)
    bins = np.logspace(0, np.log10(86400 * 100), 11)
    hist, _ = np.histogram(np.maximum(intervals, 1.0), bins=bins)
    hist_norm = hist / max(1, len(intervals))

    mean_int = _safe_float(np.mean(intervals))
    std_int = _safe_float(np.std(intervals))
    median_int = _safe_float(np.median(intervals))
    min_int = _safe_float(np.min(intervals))
    max_int = _safe_float(np.max(intervals))
    burst_ratio = _safe_float(np.sum(intervals <= 60.0) / max(1, len(intervals)))

    # Log1p scale moments to keep numerical range well-behaved
    stats_dims = [
        float(np.log1p(median_int) / 15.0),
        float(np.log1p(std_int) / 15.0),
        float(np.log1p(min_int) / 15.0),
        float(np.log1p(max_int) / 15.0),
        float(np.log1p(mean_int) / 15.0),
    ]

    vector = [float(x) for x in hist_norm] + stats_dims + [float(burst_ratio)]
    summary = {
        "median_delay_sec": round(median_int, 1),
        "mean_delay_sec": round(mean_int, 1),
        "burst_ratio": round(burst_ratio, 4),
        "min_delay_sec": round(min_int, 1),
        "max_delay_sec": round(max_int, 1)
    }
    return vector[:16], summary


def extract_gas_vector(txs: List[Transaction]) -> Tuple[List[float], Dict[str, Any]]:
    """
    Extracts 16-dimensional gas fee vector.
    - mean, std, min, max, median (5 dims)
    - quantization_score (fraction at round gwei/decimal increments) (1 dim)
    - outlier_ratio (fraction > 3 std from mean) (1 dim)
    - mode, p10, p90, skew, kurtosis (5 dims)
    - p25, p75, iqr, zero_fee_ratio (4 dims)
    """
    fees = np.array([float(tx.fee) for tx in txs if tx.fee is not None], dtype=np.float64)
    if len(fees) == 0:
        fees = np.array([0.0], dtype=np.float64)

    mean_f = _safe_float(np.mean(fees))
    std_f = _safe_float(np.std(fees))
    min_f = _safe_float(np.min(fees))
    max_f = _safe_float(np.max(fees))
    median_f = _safe_float(np.median(fees))

    # Quantization score: checking if fee aligns with round gas increments (e.g. integer gwei or round decimals)
    round_matches = 0
    for f in fees:
        # Check round decimal (0.001, 0.005, 0.01, 0.02) or integer gwei
        if abs(f * 1000 - round(f * 1000)) < 1e-4 or abs(f * 1e9 - round(f * 1e9)) < 1e-2:
            round_matches += 1
    quantization_score = float(round_matches / max(1, len(fees)))

    # Outlier ratio (> 3 std)
    outlier_threshold = mean_f + 3.0 * (std_f if std_f > 0 else 1.0)
    outlier_ratio = float(np.sum(fees > outlier_threshold) / max(1, len(fees)))

    # Percentiles and distribution shape
    p10 = _safe_float(np.percentile(fees, 10))
    p25 = _safe_float(np.percentile(fees, 25))
    p75 = _safe_float(np.percentile(fees, 75))
    p90 = _safe_float(np.percentile(fees, 90))
    iqr = _safe_float(p75 - p25)

    mode_res = stats.mode(np.round(fees, 5), keepdims=True)
    mode_f = _safe_float(mode_res.mode[0]) if len(mode_res.mode) > 0 else median_f

    if std_f < 1e-9 or np.allclose(fees, fees[0]):
        skew_f = 0.0
        kurt_f = 0.0
    else:
        skew_f = _safe_float(stats.skew(fees)) if len(fees) >= 3 else 0.0
        kurt_f = _safe_float(stats.kurtosis(fees)) if len(fees) >= 4 else 0.0
    zero_ratio = float(np.sum(fees == 0.0) / max(1, len(fees)))

    vector = [
        float(mean_f),
        float(std_f),
        float(min_f),
        float(max_f),
        float(median_f),
        float(quantization_score),
        float(outlier_ratio),
        float(mode_f),
        float(p10),
        float(p90),
        float(np.clip(skew_f, -5.0, 5.0) / 5.0),
        float(np.clip(kurt_f, -5.0, 10.0) / 10.0),
        float(p25),
        float(p75),
        float(iqr),
        float(zero_ratio)
    ]

    summary = {
        "mean_fee": round(mean_f, 6),
        "median_fee": round(median_f, 6),
        "quantization_score": round(quantization_score, 4),
        "outlier_ratio": round(outlier_ratio, 4)
    }
    return vector[:16], summary


def extract_amount_vector(
    txs: List[Transaction], in_txs: Optional[List[Transaction]] = None
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Extracts 16-dimensional amount structuring vector.
    - mean, std, median, log_mean (4 dims)
    - round_fraction: fraction divisible by 10/100/1000 or clean round decimals (1 dim)
    - top-5 most common denomination brackets (<0.1, 0.1-1, 1-10, 10-100, >100) (5 dims)
    - split_ratio_mean: out_amount / in_amount (1 dim)
    - min, max, p25, p75, skew (5 dims)
    """
    amounts = np.array([float(tx.amount) for tx in txs], dtype=np.float64)
    if len(amounts) == 0:
        amounts = np.array([0.0], dtype=np.float64)

    mean_a = _safe_float(np.mean(amounts))
    std_a = _safe_float(np.std(amounts))
    median_a = _safe_float(np.median(amounts))
    log_mean = _safe_float(np.log1p(mean_a) / 10.0)

    # Round number structuring detection
    round_count = 0
    for a in amounts:
        if a > 0 and (
            a % 10 == 0 or a % 100 == 0 or a % 1000 == 0 or
            abs(a - round(a, 1)) < 1e-4 or abs(a - round(a, 0)) < 1e-4
        ):
            round_count += 1
    round_fraction = float(round_count / max(1, len(amounts)))

    # 5-tier denomination ratios
    d1 = float(np.sum(amounts < 0.1) / len(amounts))
    d2 = float(np.sum((amounts >= 0.1) & (amounts < 1.0)) / len(amounts))
    d3 = float(np.sum((amounts >= 1.0) & (amounts < 10.0)) / len(amounts))
    d4 = float(np.sum((amounts >= 10.0) & (amounts < 100.0)) / len(amounts))
    d5 = float(np.sum(amounts >= 100.0) / len(amounts))

    # Split ratio: out_amount / in_amount
    out_total = float(np.sum(amounts))
    in_total = float(np.sum([t.amount for t in in_txs])) if in_txs else out_total
    split_ratio_mean = float(out_total / max(1e-4, in_total))
    split_ratio_clamped = float(np.clip(split_ratio_mean, 0.0, 5.0) / 5.0)
    min_a = _safe_float(np.min(amounts))
    max_a = _safe_float(np.max(amounts))
    p25_a = _safe_float(np.percentile(amounts, 25))
    p75_a = _safe_float(np.percentile(amounts, 75))

    if std_a < 1e-9 or np.allclose(amounts, amounts[0]):
        skew_a = 0.0
    else:
        skew_a = _safe_float(stats.skew(amounts)) if len(amounts) >= 3 else 0.0

    vector = [
        float(np.log1p(mean_a) / 10.0),
        float(np.log1p(std_a) / 10.0),
        float(np.log1p(median_a) / 10.0),
        float(log_mean),
        float(round_fraction),
        float(d1),
        float(d2),
        float(d3),
        float(d4),
        float(d5),
        float(split_ratio_clamped),
        float(np.log1p(min_a) / 10.0),
        float(np.log1p(max_a) / 10.0),
        float(np.log1p(p25_a) / 10.0),
        float(np.log1p(p75_a) / 10.0),
        float(np.clip(skew_a, -5.0, 5.0) / 5.0)
    ]

    summary = {
        "mean_amount": round(mean_a, 4),
        "median_amount": round(median_a, 4),
        "round_fraction": round(round_fraction, 4),
        "split_ratio": round(split_ratio_mean, 4)
    }
    return vector[:16], summary


def extract_chain_temporal_vector(txs: List[Transaction]) -> Tuple[List[float], Dict[str, Any]]:
    """
    Extracts 16-dimensional chain and temporal vector.
    - fraction of txs per chain (BTC, ETH, TRON_TRC20) (3 dims)
    - days_active (1 dim)
    - avg_tx_per_day (1 dim)
    - unique_hour_of_day_histogram (8 bins across 24 hours: 3h intervals) (8 dims)
    - weekend_ratio, night_ratio, is_cross_chain_user (3 dims)
    """
    chains = [tx.chain.value if hasattr(tx.chain, "value") else str(tx.chain) for tx in txs]
    total_tx = max(1, len(chains))
    frac_btc = float(sum(1 for c in chains if "btc" in c.lower()) / total_tx)
    frac_eth = float(sum(1 for c in chains if "eth" in c.lower()) / total_tx)
    frac_tron = float(sum(1 for c in chains if "tron" in c.lower() or "trc" in c.lower()) / total_tx)

    timestamps = [tx.timestamp for tx in txs]
    t_min = min(timestamps) if timestamps else 0
    t_max = max(timestamps) if timestamps else 0
    days_active = max(1.0, (t_max - t_min) / 86400.0)
    avg_tx_per_day = float(len(txs) / days_active)

    # Hour of day (UTC)
    hours = [(t % 86400) // 3600 for t in timestamps]
    # 8 bins of 3 hours each: [0-3, 3-6, 6-9, 9-12, 12-15, 15-18, 18-21, 21-24]
    hour_bins = np.zeros(8, dtype=np.float64)
    night_count = 0
    for h in hours:
        b_idx = min(7, int(h // 3))
        hour_bins[b_idx] += 1
        if 0 <= h < 6:
            night_count += 1
    hour_bins_norm = hour_bins / max(1, len(hours))
    night_ratio = float(night_count / max(1, len(hours)))

    # Day of week (Unix epoch 0 is Thursday, Jan 1 1970)
    # Days since epoch = t // 86400; (day + 4) % 7 gives 0=Sunday, 6=Saturday
    weekend_count = 0
    for t in timestamps:
        dow = ((t // 86400) + 4) % 7
        if dow in (0, 6):
            weekend_count += 1
    weekend_ratio = float(weekend_count / max(1, len(timestamps)))

    unique_chains = set(chains)
    is_cross_chain = 1.0 if len(unique_chains) > 1 else 0.0

    vector = [
        float(frac_btc),
        float(frac_eth),
        float(frac_tron),
        float(np.log1p(days_active) / 5.0),
        float(np.log1p(avg_tx_per_day) / 5.0),
        float(hour_bins_norm[0]),
        float(hour_bins_norm[1]),
        float(hour_bins_norm[2]),
        float(hour_bins_norm[3]),
        float(hour_bins_norm[4]),
        float(hour_bins_norm[5]),
        float(hour_bins_norm[6]),
        float(hour_bins_norm[7]),
        float(weekend_ratio),
        float(night_ratio),
        float(is_cross_chain)
    ]

    summary = {
        "chains": list(unique_chains),
        "days_active": round(days_active, 1),
        "weekend_ratio": round(weekend_ratio, 4),
        "night_ratio": round(night_ratio, 4),
        "is_cross_chain": bool(is_cross_chain)
    }
    return vector[:16], summary


def extract_fingerprint(
    store: NetworkXStore,
    address: str,
    min_txs: int = 3,
    txs: Optional[List[Transaction]] = None,
    strict: bool = False,
    direction: str = "out"
) -> BehavioralFingerprint:
    """
    Extracts 64-dimensional behavioral fingerprint for a specific wallet address.
    If the wallet has fewer than min_txs outgoing transactions (< 3), returns
    a fingerprint with summary flag "confidence": "INSUFFICIENT_DATA" (unless strict=True).
    """
    if txs is None:
        txs = []
        if address in store.graph:
            if direction in ("out", "both"):
                for _, _, data in store.graph.out_edges(address, data=True):
                    tx_hash = data.get("tx_hash")
                    if tx_hash and tx_hash in store.transactions:
                        txs.append(store.transactions[tx_hash])
            if direction in ("in", "both"):
                for _, _, data in store.graph.in_edges(address, data=True):
                    tx_hash = data.get("tx_hash")
                    if tx_hash and tx_hash in store.transactions and store.transactions[tx_hash] not in txs:
                        txs.append(store.transactions[tx_hash])

    if strict and len(txs) < min_txs:
        raise ValueError(
            f"Wallet '{address}' has only {len(txs)} transactions; minimum {min_txs} required."
        )

    # Fetch incoming transactions for flow ratio calculation
    in_txs = []
    if address in store.graph:
        for _, _, data in store.graph.in_edges(address, data=True):
            tx_hash = data.get("tx_hash")
            if tx_hash and tx_hash in store.transactions:
                in_txs.append(store.transactions[tx_hash])

    timing_vec, timing_sum = extract_timing_vector(txs)
    gas_vec, gas_sum = extract_gas_vector(txs)
    amount_vec, amount_sum = extract_amount_vector(txs, in_txs)
    chain_vec, chain_sum = extract_chain_temporal_vector(txs)

    # Determine confidence status based on transaction maturity threshold
    is_insufficient = len(txs) < min_txs

    combined_summary = {
        **timing_sum,
        **gas_sum,
        **amount_sum,
        **chain_sum,
        "total_tx_analyzed": len(txs),
        "confidence": "INSUFFICIENT_DATA" if is_insufficient else "SUFFICIENT_DATA",
        "maturity_bucket": "Bucket A (1-2 txs)" if len(txs) <= 2 else ("Bucket B (3-5 txs)" if len(txs) <= 5 else "Bucket C (6+ txs)")
    }

    wallet_obj = store.get_wallet(address)
    chain_str = wallet_obj.chain.value if wallet_obj and hasattr(wallet_obj.chain, "value") else "ETH"

    return BehavioralFingerprint(
        address=address,
        chain=chain_str,
        tx_count=len(txs),
        timing_vector=timing_vec,
        gas_vector=gas_vec,
        amount_vector=amount_vec,
        chain_vector=chain_vec,
        summary=combined_summary
    )


def extract_fingerprint_safe(
    store: NetworkXStore,
    address: str,
    fallback_path: Optional[List[str]] = None,
    direction: str = "out"
) -> Optional[BehavioralFingerprint]:
    """
    Safe extraction wrapper that extracts the wallet's behavioral fingerprint.
    If the wallet has fewer than 3 outgoing transactions, the fingerprint will contain
    the summary flag 'confidence': 'INSUFFICIENT_DATA'.
    No artificial path aggregation is performed to fake maturity for suspect wallets.
    """
    if address not in store.graph and address not in store.wallets:
        return None
    return extract_fingerprint(store, address, min_txs=3, direction=direction)
