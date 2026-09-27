import sys
sys.path.insert(0, '.')
from app.data.synthetic_generator import SyntheticDataGenerator
from app.config import get_settings
from app.graph.networkx_store import NetworkXStore

settings = get_settings()
store = NetworkXStore()
gen = SyntheticDataGenerator(seed=settings.RANDOM_SEED)
vasps, wallets, transactions, test_cases = gen.generate()
gen.populate_store(store)

print("=================================================================")
print("CHECK 1: VASPS, SAMPLE WALLETS, AND TRANSACTIONS")
print("=================================================================")
print("=== VASPs ===")
for v in vasps:
    chains = sorted(list({w.chain.value for w in wallets if w.vasp_id == v.id}))
    print(f"  {v.name:22s} | {v.category.value:25s} | Chains: {', '.join(chains)}")

print("\n=== Sample 5 Wallets ===")
for w in wallets[:5]:
    vasp_name = next((v.name for v in vasps if v.id == w.vasp_id), "None")
    lbl = w.label or "Unlabeled"
    print(f"  {w.address[:22]}... | chain={w.chain.value:11s} | true_vasp={vasp_name:16s} | label={lbl}")

print("\n=== Sample 5 Transactions ===")
for tx in transactions[:5]:
    pat = "peeling" if tx.is_peeling_chain else ("sweeper" if tx.is_sweeper_tx else ("mixer" if tx.is_mixer_tx else "standard"))
    print(f"  {tx.tx_hash[:18]}... | {tx.chain.value:11s} | {tx.from_address[:14]}... -> {tx.to_address[:14]}... | {tx.amount:>10.2f} {tx.token_symbol:5s} | pattern={pat}")

print(f"\nTotal: {len(wallets)} wallets, {len(transactions)} txs, {len(test_cases)} cases")

print("\n=================================================================")
print("CHECK 2: TRACE CASE-001 END-TO-END")
print("=================================================================")
c1 = next(c for c in test_cases if c.case_id == "CASE-001")
candidates = store.find_nearest_vasp(c1.suspect_wallet, max_hops=6)
print(f"Case ID:        {c1.case_id}")
print(f"Description:    {c1.case_description}")
print(f"Suspect Wallet: {c1.suspect_wallet}")
print(f"Expected VASP:  {c1.expected_vasp}")
if candidates:
    top = candidates[0]
    target_wallet = top["target_wallet"]
    vasp_name = top["vasp_name"]
    print(f"Top Target Hit: {target_wallet} (VASP: {vasp_name})")
    print(f"Total Hops:     {top['proximity_rank']}")
    print("Hops Sequence:")
    for i in range(len(top["path"]) - 1):
        u = top["path"][i]
        v = top["path"][i+1]
        tx_h = top["tx_hashes"][i]
        tx_obj = store.transactions.get(tx_h)
        amt = f"{tx_obj.amount} {tx_obj.token_symbol}" if tx_obj else "N/A"
        print(f"  Hop {i+1}: {u} ->\n         {v}\n         Tx: {tx_h} | Amount: {amt}")
else:
    print("No path found!")

print("\n=================================================================")
print("CHECK 3: BASELINE PROXIMITY ACCURACY (Pure Graph Distance)")
print("=================================================================")
correct_count = 0
print(f"{'Case ID':<10} | {'Expected VASP':<18} | {'Predicted VASP':<18} | {'Hops':<5} | {'Match?':<6}")
print("-" * 65)
for case in test_cases:
    res = store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
    if res:
        pred_vasp = res[0]["vasp_id"]
        hops = res[0]["proximity_rank"]
    else:
        pred_vasp = "NONE"
        hops = -1
    match = "Y" if pred_vasp == case.expected_vasp else "N"
    if match == "Y":
        correct_count += 1
    print(f"{case.case_id:<10} | {case.expected_vasp:<18} | {pred_vasp:<18} | {hops:<5} | {match:<6}")

print(f"\nBaseline Accuracy: {correct_count}/{len(test_cases)} ({correct_count/len(test_cases)*100:.1f}%)")

print("\n=================================================================")
print("CHECK 4: CROSS-CHAIN TEST CASES ANALYSIS")
print("=================================================================")
cross_chain_count = 0
for case in test_cases:
    # Find path to expected VASP
    candidates = store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
    expected_candidate = next((c for c in candidates if c["vasp_id"] == case.expected_vasp), None)
    
    if expected_candidate:
        path = expected_candidate["path"]
        chains_in_path = [store.wallets[addr].chain.value for addr in path if addr in store.wallets]
        unique_chains = list(dict.fromkeys(chains_in_path))
        is_cross = len(unique_chains) > 1 or case.is_cross_chain
    else:
        unique_chains = [case.chain.value]
        is_cross = case.is_cross_chain

    if is_cross:
        cross_chain_count += 1
    status = "YES" if is_cross else "NO"
    chain_str = " -> ".join(unique_chains)
    print(f"{case.case_id}: Primary = {case.chain.value:<11s} | Expected Path Chains = {chain_str:<32s} | Cross-Chain: {status}")

print(f"\nTotal Cross-Chain Cases: {cross_chain_count}/{len(test_cases)}")
