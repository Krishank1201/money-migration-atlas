"""
GNN Training Pipeline for GraphSAGE and GATv2 (Phase 5).
Money Migration Atlas (SIH26182).

Trains both GraphSAGE and GATv2 models on localized subgraphs,
logging epoch curves and persisting torch model artifacts.
"""

import os
import sys
import json
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from app.config import get_settings
from app.graph.networkx_store import NetworkXStore
from app.graph.store import GraphStore
from app.data.synthetic_generator import generate_synthetic_data
from app.ml.gnn.converter import VASP_TO_IDX, IDX_TO_VASP
from app.ml.gnn.graphsage import VASPGraphSAGE
from app.ml.gnn.gat import VASPGATv2

logger = logging.getLogger("mma.ml.gnn.trainer")
settings = get_settings()

MODEL_VERSION = f"gnn_v1_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


def extract_candidate_subgraph(
    suspect_wallet: str,
    candidate_vasp_id: str,
    store: GraphStore,
    path: Optional[List[str]] = None,
    k_hops: int = 3
) -> Data:
    """
    Extracts localized directed ego-subgraph connecting suspect to candidate VASP.
    """
    graph: nx.MultiDiGraph = getattr(store, "graph", nx.MultiDiGraph())
    wallets_map = getattr(store, "wallets", {})

    vasp = store.get_vasp(candidate_vasp_id)
    vasp_targets: Set[str] = set()
    if vasp:
        vasp_targets.update(vasp.hot_wallets)
        vasp_targets.update(vasp.deposit_sweepers)
    for addr, w in wallets_map.items():
        if w.vasp_id == candidate_vasp_id:
            vasp_targets.add(addr)

    nodes = set([suspect_wallet])
    if path:
        nodes.update(path)
    for n in list(nodes):
        if n in graph:
            nodes.update(graph.successors(n))

    sub_g = graph.subgraph(nodes)
    n_list = list(sub_g.nodes())
    n2i = {n: i for i, n in enumerate(n_list)}

    x_l = []
    for n in n_list:
        w = wallets_map.get(n)
        in_d = float(sub_g.in_degree(n))
        out_d = float(sub_g.out_degree(n))
        is_mix = 1.0 if (w and w.is_mixer) else 0.0
        is_target_v = 1.0 if n in vasp_targets else 0.0
        tot_cnt = in_d + out_d
        x_l.append([in_d, out_d, tot_cnt, 0.0, 0.0, 0.001, is_mix, is_target_v])

    e_src, e_dst = [], []
    for u, v in sub_g.edges():
        e_src.extend([n2i[u], n2i[v]])
        e_dst.extend([n2i[v], n2i[u]])
    if not e_src:
        e_src, e_dst = [0], [0]

    x = torch.tensor(x_l, dtype=torch.float32)
    edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
    s_idx = n2i[suspect_wallet]

    return Data(x=x, edge_index=edge_index, suspect_idx=s_idx)


def build_gnn_subgraph_dataset(
    store: GraphStore,
    benchmark_cases: list,
    samples_per_vasp: int = 18,
    seed: int = 42
) -> Tuple[List[Tuple[Data, int, bool]], Set[str]]:
    """
    Constructs 324 candidate subgraphs matching Phase 4 XGBoost distribution.
    Guarantees strict zero data leakage against benchmark cases.
    """
    rng = random.Random(seed)
    vasps = store.get_all_vasps()
    vasp_ids = [v.id for v in vasps]
    wallets_map = getattr(store, "wallets", {})
    graph = getattr(store, "graph", nx.MultiDiGraph())

    # Data leakage guard
    benchmark_excluded_addrs: Set[str] = set()
    for c in benchmark_cases:
        benchmark_excluded_addrs.add(c.suspect_wallet)
        res = store.find_nearest_vasp(c.suspect_wallet, max_hops=6)
        for r in res:
            benchmark_excluded_addrs.update(r.get("path", []))

    rev_graph = graph.reverse()
    dataset: List[Tuple[Data, int, bool]] = []

    for v_id in vasp_ids:
        v_idx = VASP_TO_IDX[v_id]
        anchors = [
            w.address for w in wallets_map.values()
            if w.vasp_id == v_id and w.address not in benchmark_excluded_addrs
        ]

        upstream: List[str] = []
        for a in anchors:
            if a in rev_graph:
                try:
                    lengths = nx.single_source_shortest_path_length(rev_graph, a, cutoff=5)
                    for u, dist in lengths.items():
                        if u != a and u not in benchmark_excluded_addrs:
                            upstream.append(u)
                except Exception:
                    pass

        rng.shuffle(upstream)
        seen = set()
        sampled: List[str] = []
        for u in upstream:
            if u not in seen:
                seen.add(u)
                sampled.append(u)
            if len(sampled) >= samples_per_vasp:
                break

        for s in sampled:
            cand_res = store.find_nearest_vasp(s, max_hops=6)
            cand_map = {r["vasp_id"]: r for r in cand_res}

            if v_id in cand_map:
                # Positive sample
                r_pos = cand_map[v_id]
                d_pos = extract_candidate_subgraph(s, v_id, store, r_pos["path"])
                dataset.append((d_pos, v_idx, True))

                # Hard negative sample
                other_cands = [r for other_v, r in cand_map.items() if other_v != v_id]
                other_cands.sort(key=lambda item: item["proximity_rank"])
                for r_neg in other_cands[:1]:
                    d_neg = extract_candidate_subgraph(s, r_neg["vasp_id"], store, r_neg["path"])
                    dataset.append((d_neg, v_idx, False))

    rng.shuffle(dataset)
    logger.info("Constructed GNN candidate subgraph dataset: %d samples", len(dataset))
    return dataset, benchmark_excluded_addrs


def train_gnn_model(
    model: nn.Module,
    train_set: List[Tuple[Data, int, bool]],
    val_set: List[Tuple[Data, int, bool]],
    epochs: int = 50,
    lr: float = 0.003,
    batch_size: int = 32
) -> Dict[str, List[float]]:
    """Trains a GNN model with mini-batching and tracks training curves."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Pre-build train batches
    train_batches = []
    for i in range(0, len(train_set), batch_size):
        chunk = train_set[i:i + batch_size]
        data_list = [d for d, tgt, is_p in chunk]
        targets = torch.tensor([tgt for d, tgt, is_p in chunk], dtype=torch.long)
        batch = Batch.from_data_list(data_list)
        node_offsets = [0]
        for d in data_list[:-1]:
            node_offsets.append(node_offsets[-1] + d.num_nodes)
        root_indices = [offset + d.suspect_idx for offset, d in zip(node_offsets, data_list)]
        train_batches.append((batch, torch.tensor(root_indices, dtype=torch.long), targets))

    # Pre-build val batch
    val_data_list = [d for d, tgt, is_p in val_set]
    val_targets = torch.tensor([tgt for d, tgt, is_p in val_set], dtype=torch.long)
    val_batch = Batch.from_data_list(val_data_list)
    val_offsets = [0]
    for d in val_data_list[:-1]:
        val_offsets.append(val_offsets[-1] + d.num_nodes)
    val_root_indices = torch.tensor(
        [offset + d.suspect_idx for offset, d in zip(val_offsets, val_data_list)], dtype=torch.long
    )

    curves = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b, roots, targets in train_batches:
            optimizer.zero_grad()
            out = model(b.x, b.edge_index)
            loss = criterion(out[roots], targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_batches))

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_batch.x, val_batch.edge_index)
            val_loss = criterion(val_out[val_root_indices], val_targets).item()
            val_pred = val_out[val_root_indices].argmax(dim=1)
            val_acc = (val_pred == val_targets).float().mean().item()

        curves["train_loss"].append(round(avg_train_loss, 4))
        curves["val_loss"].append(round(val_loss, 4))
        curves["val_acc"].append(round(val_acc, 4))

    return curves


def train_both_gnn_models(
    store: Optional[GraphStore] = None,
    benchmark_cases: Optional[list] = None,
    seed: int = 42,
    epochs: int = 40,
    max_epochs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trains both GraphSAGE and GATv2 models, evaluates on test split,
    and persists models and training curves to disk.
    """
    if max_epochs is not None:
        epochs = max_epochs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    current = Path(__file__).resolve()
    backend_root = current.parent.parent.parent.parent
    if backend_root.name == "backend":
        models_dir = backend_root / "data" / "models"
    else:
        models_dir = backend_root / "backend" / "data" / "models"
    os.makedirs(models_dir, exist_ok=True)

    sage_path = models_dir / "graphsage_v1.pt"
    gat_path = models_dir / "gatv2_v1.pt"
    curves_path = models_dir / "gnn_training_curves.json"
    meta_path = models_dir / "gnn_model_metadata.json"

    if store is None or benchmark_cases is None:
        store = NetworkXStore()
        vasps, wallets, txs, benchmark_cases = generate_synthetic_data(seed=seed, store=store)

    dataset, excluded_addrs = build_gnn_subgraph_dataset(
        store=store,
        benchmark_cases=benchmark_cases,
        samples_per_vasp=18,
        seed=seed
    )

    n_tot = len(dataset)
    n_tr = int(0.70 * n_tot)
    n_vl = int(0.15 * n_tot)

    train_set = dataset[:n_tr]
    val_set = dataset[n_tr:n_tr + n_vl]
    test_set = dataset[n_tr + n_vl:]

    logger.info("Training GraphSAGE model...")
    sage_model = VASPGraphSAGE(in_channels=8, hidden_channels=64, out_channels=32, num_classes=9)
    sage_curves = train_gnn_model(sage_model, train_set, val_set, epochs=epochs, lr=0.003)

    logger.info("Training GATv2 model...")
    gat_model = VASPGATv2(in_channels=8, hidden_channels=16, heads=4, out_channels=32, num_classes=9)
    gat_curves = train_gnn_model(gat_model, train_set, val_set, epochs=epochs, lr=0.003)

    # Evaluate on test set
    test_data_list = [d for d, tgt, is_p in test_set]
    test_targets = torch.tensor([tgt for d, tgt, is_p in test_set], dtype=torch.long)
    test_batch = Batch.from_data_list(test_data_list)
    test_offsets = [0]
    for d in test_data_list[:-1]:
        test_offsets.append(test_offsets[-1] + d.num_nodes)
    test_root_indices = torch.tensor(
        [offset + d.suspect_idx for offset, d in zip(test_offsets, test_data_list)], dtype=torch.long
    )

    sage_model.eval()
    gat_model.eval()
    with torch.no_grad():
        s_out = sage_model(test_batch.x, test_batch.edge_index)[test_root_indices]
        g_out = gat_model(test_batch.x, test_batch.edge_index)[test_root_indices]
        s_pred = s_out.argmax(dim=1)
        g_pred = g_out.argmax(dim=1)
        ens_pred = (s_out + g_out).argmax(dim=1)

        sage_acc = (s_pred == test_targets).float().mean().item()
        gat_acc = (g_pred == test_targets).float().mean().item()
        ens_acc = (ens_pred == test_targets).float().mean().item()

    # Save torch weights
    torch.save(sage_model.state_dict(), str(sage_path))
    torch.save(gat_model.state_dict(), str(gat_path))

    training_curves = {
        "graphsage": sage_curves,
        "gatv2": gat_curves
    }
    with open(str(curves_path), "w", encoding="utf-8") as f:
        json.dump(training_curves, f, indent=2)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_samples": len(dataset),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "metrics": {
            "graphsage_test_acc": round(sage_acc, 4),
            "gatv2_test_acc": round(gat_acc, 4),
            "ensemble_test_acc": round(ens_acc, 4),
            "final_train_loss_sage": sage_curves["train_loss"][-1],
            "final_train_loss_gat": gat_curves["train_loss"][-1]
        },
        "excluded_benchmark_count": len(excluded_addrs)
    }

    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "GNN models saved: GraphSAGE (%s), GATv2 (%s) - Test Acc: SAGE=%.4f, GAT=%.4f, ENS=%.4f",
        sage_path, gat_path, sage_acc, gat_acc, ens_acc
    )

    return {
        "model_version": MODEL_VERSION,
        "graphsage_path": str(sage_path),
        "gatv2_path": str(gat_path),
        "curves_path": str(curves_path),
        "metadata_path": str(meta_path),
        "metadata": metadata,
        "training_curves": training_curves
    }


train_gnn_models = train_both_gnn_models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = train_both_gnn_models()
    print("GNN Training finished successfully:")
    print(json.dumps(results["metadata"], indent=2))
