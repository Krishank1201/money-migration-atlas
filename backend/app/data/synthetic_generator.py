import json
import random
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

from app.core.schemas import (
    Chain,
    VASP,
    VASPCategory,
    Wallet,
    Transaction,
    GroundTruthTestCase,
    LaunderingPattern
)
from app.graph.store import GraphStore


# Seed VASPs (FIU-IND Indian and Overseas)
SEED_VASPS: List[Dict[str, Any]] = [
    {
        "id": "coindcx",
        "name": "CoinDCX",
        "legal_name": "Primestack Pte Ltd / Neblio Technologies Pvt Ltd",
        "category": VASPCategory.INDIAN_FIU,
        "jurisdiction": "IN",
        "fiu_ind_registered": True,
        "fiu_registration_number": "FIU-IND-VASP-001",
    },
    {
        "id": "wazirx",
        "name": "WazirX",
        "legal_name": "Zanmai Labs Pvt Ltd",
        "category": VASPCategory.INDIAN_FIU,
        "jurisdiction": "IN",
        "fiu_ind_registered": True,
        "fiu_registration_number": "FIU-IND-VASP-002",
    },
    {
        "id": "zebpay",
        "name": "ZebPay",
        "legal_name": "Awlencan Innovations India Pvt Ltd",
        "category": VASPCategory.INDIAN_FIU,
        "jurisdiction": "IN",
        "fiu_ind_registered": True,
        "fiu_registration_number": "FIU-IND-VASP-003",
    },
    {
        "id": "coinswitch",
        "name": "CoinSwitch",
        "legal_name": "Bitcipher Labs LLP",
        "category": VASPCategory.INDIAN_FIU,
        "jurisdiction": "IN",
        "fiu_ind_registered": True,
        "fiu_registration_number": "FIU-IND-VASP-004",
    },
    {
        "id": "mudrex",
        "name": "Mudrex",
        "legal_name": "Mudrex Technologies Pvt Ltd",
        "category": VASPCategory.INDIAN_FIU,
        "jurisdiction": "IN",
        "fiu_ind_registered": True,
        "fiu_registration_number": "FIU-IND-VASP-005",
    },
    {
        "id": "unocoin",
        "name": "Unocoin",
        "legal_name": "Unocoin Technologies Pvt Ltd",
        "category": VASPCategory.INDIAN_FIU,
        "jurisdiction": "IN",
        "fiu_ind_registered": True,
        "fiu_registration_number": "FIU-IND-VASP-006",
    },
    {
        "id": "giottus",
        "name": "Giottus",
        "legal_name": "Giottus Technologies Pvt Ltd",
        "category": VASPCategory.INDIAN_FIU,
        "jurisdiction": "IN",
        "fiu_ind_registered": True,
        "fiu_registration_number": "FIU-IND-VASP-007",
    },
    {
        "id": "binance_offshore",
        "name": "Binance (Offshore)",
        "legal_name": "Binance Holdings Limited",
        "category": VASPCategory.OVERSEAS_NON_COMPLIANT,
        "jurisdiction": "KY",
        "fiu_ind_registered": False,
        "fiu_registration_number": None,
    },
    {
        "id": "bybit",
        "name": "Bybit",
        "legal_name": "Bybit Fintech Limited",
        "category": VASPCategory.OVERSEAS_NON_COMPLIANT,
        "jurisdiction": "AE",
        "fiu_ind_registered": False,
        "fiu_registration_number": None,
    }
]


class SyntheticDataGenerator:
    """
    Generates realistic, deterministic multi-chain cryptocurrency transaction graphs
    with simulated criminal money laundering topologies.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.wallets: Dict[str, Wallet] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.vasps: Dict[str, VASP] = {}
        self.test_cases: List[GroundTruthTestCase] = []
        self.benchmark_addresses: set = set()

    def _rand_hex(self, length: int) -> str:
        return "".join(self.rng.choices("0123456789abcdef", k=length))

    def _generate_address(self, chain: Chain, tag: str = "") -> str:
        entropy = f"{tag}_{self.rng.randint(100000, 99999999)}_{self._rand_hex(8)}"
        h = hashlib.sha256(entropy.encode()).hexdigest()
        if chain == Chain.BTC:
            return f"bc1q{h[:38]}"
        elif chain == Chain.ETH:
            return f"0x{h[:40]}"
        elif chain == Chain.TRON_TRC20:
            b58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            return "T" + "".join(self.rng.choices(b58, k=33))
        else:
            return f"0x{h[:40]}"

    def _generate_tx_hash(self, chain: Chain) -> str:
        raw = self._rand_hex(64)
        return f"0x{raw}" if chain in (Chain.ETH, Chain.POLYGON, Chain.BSC) else raw

    def _register_benchmark_wallet(self, address: str, chain: Chain, label: str, risk: float = 0.5, is_mixer: bool = False) -> str:
        self.wallets[address] = Wallet(
            address=address,
            chain=chain,
            label=label,
            risk_score=risk,
            is_mixer=is_mixer
        )
        self.benchmark_addresses.add(address)
        return address

    def _get_vasp_wallet(self, vasp_id: str, chain: Chain, is_hot: bool = True) -> str:
        for w in self.wallets.values():
            if w.vasp_id == vasp_id and w.chain == chain:
                if is_hot and w.is_vasp_hot_wallet:
                    return w.address
                elif not is_hot and w.is_vasp_deposit_sweeper:
                    return w.address
        # Fallback to any wallet of that VASP
        for w in self.wallets.values():
            if w.vasp_id == vasp_id:
                return w.address
        raise ValueError(f"VASP wallet not found for {vasp_id} on {chain}")

    # -------------------------------------------------------------------------
    # Reusable Topology Helpers
    # -------------------------------------------------------------------------

    def _make_cross_chain_bridge_path(
        self,
        source_wallet: str,
        target_vasp: str,
        chains: List[Chain] = [Chain.BTC, Chain.ETH, Chain.TRON_TRC20],
        tag_prefix: str = "bridge"
    ) -> List[Transaction]:
        """
        Constructs a realistic multi-chain bridge corridor.
        Swaps value across consecutive chains and terminates at target_vasp on chains[-1].
        """
        txs: List[Transaction] = []
        current_node = source_wallet
        base_time = 1711000000

        token_units = {Chain.BTC: ("BTC", 2.5), Chain.ETH: ("ETH", 40.0), Chain.TRON_TRC20: ("USDT", 100000.0)}

        for i in range(len(chains) - 1):
            src_chain = chains[i]
            dst_chain = chains[i + 1]

            # Bridge deposit contract on src_chain
            bridge_dep = self._generate_address(src_chain, tag=f"{tag_prefix}_dep_{i}_{src_chain.value}")
            self._register_benchmark_wallet(bridge_dep, src_chain, f"Cross-Chain Bridge Deposit ({src_chain.value})", risk=0.2)

            tok_src, amt_src = token_units.get(src_chain, ("TOKEN", 10.0))
            t1 = Transaction(
                tx_hash=self._generate_tx_hash(src_chain),
                chain=src_chain,
                from_address=current_node,
                to_address=bridge_dep,
                amount=amt_src,
                token_symbol=tok_src,
                timestamp=base_time + (i * 3600),
                fee=0.001
            )
            self.transactions[t1.tx_hash] = t1
            txs.append(t1)

            # Bridge release / router node on dst_chain
            bridge_out = self._generate_address(dst_chain, tag=f"{tag_prefix}_out_{i}_{dst_chain.value}")
            self._register_benchmark_wallet(bridge_out, dst_chain, f"Cross-Chain Bridge Release ({dst_chain.value})", risk=0.2)

            # Cross-chain synthetic swap transaction linking the bridge endpoints
            tok_dst, amt_dst = token_units.get(dst_chain, ("TOKEN", 10.0))
            t_bridge = Transaction(
                tx_hash=self._generate_tx_hash(dst_chain),
                chain=dst_chain,
                from_address=bridge_dep,
                to_address=bridge_out,
                amount=amt_dst,
                token_symbol=tok_dst,
                timestamp=base_time + (i * 3600) + 600,
                fee=0.002
            )
            self.transactions[t_bridge.tx_hash] = t_bridge
            txs.append(t_bridge)

            current_node = bridge_out

        # Final hop into target VASP
        final_chain = chains[-1]
        target_hw = self._get_vasp_wallet(target_vasp, final_chain, is_hot=True)
        tok_final, amt_final = token_units.get(final_chain, ("TOKEN", 10.0))

        t_final = Transaction(
            tx_hash=self._generate_tx_hash(final_chain),
            chain=final_chain,
            from_address=current_node,
            to_address=target_hw,
            amount=amt_final * 0.99,
            token_symbol=tok_final,
            timestamp=base_time + (len(chains) * 3600),
            fee=0.001
        )
        self.transactions[t_final.tx_hash] = t_final
        txs.append(t_final)

        return txs

    def _make_proximity_tie_path(
        self,
        source_wallet: str,
        vasp_a: str,
        vasp_b: str,
        hops: int = 3,
        chain: Chain = Chain.BTC,
        tag_prefix: str = "tie"
    ) -> None:
        """
        Creates two equal-length paths (same hop count) from source_wallet:
        - Path A (clean path) -> vasp_a hot wallet
        - Path B (decoy path touching a mixer with larger volume) -> vasp_b hot wallet
        Because Path B has higher volume (lower Dijkstra weight), naive proximity ranks vasp_b #1.
        """
        base_time = 1711200000

        # Path A (Clean Path -> vasp_a)
        curr_a = source_wallet
        for h in range(hops - 1):
            next_a = self._generate_address(chain, tag=f"{tag_prefix}_clean_h{h}")
            self._register_benchmark_wallet(next_a, chain, f"Clean Transit {h+1}", risk=0.25)
            tx = Transaction(
                tx_hash=self._generate_tx_hash(chain),
                chain=chain,
                from_address=curr_a,
                to_address=next_a,
                amount=5.0,
                token_symbol="BTC",
                timestamp=base_time + (h * 900),
                fee=0.0003
            )
            self.transactions[tx.tx_hash] = tx
            curr_a = next_a

        vasp_a_hw = self._get_vasp_wallet(vasp_a, chain, is_hot=True)
        tx_a_final = Transaction(
            tx_hash=self._generate_tx_hash(chain),
            chain=chain,
            from_address=curr_a,
            to_address=vasp_a_hw,
            amount=4.8,
            token_symbol="BTC",
            timestamp=base_time + (hops * 900),
            fee=0.0003
        )
        self.transactions[tx_a_final.tx_hash] = tx_a_final

        # Path B (Decoy Path -> vasp_b via mixer with higher amount, giving lower Dijkstra distance)
        curr_b = source_wallet
        for h in range(hops - 1):
            is_mix = (h == 0)
            next_b = self._generate_address(chain, tag=f"{tag_prefix}_decoy_h{h}")
            lbl = "Mixer Decoy Pool" if is_mix else f"Decoy Transit {h+1}"
            self._register_benchmark_wallet(next_b, chain, lbl, risk=0.95 if is_mix else 0.7, is_mixer=is_mix)
            tx = Transaction(
                tx_hash=self._generate_tx_hash(chain),
                chain=chain,
                from_address=curr_b,
                to_address=next_b,
                amount=75.0,  # Higher volume ranks it first in weighted distance
                token_symbol="BTC",
                timestamp=base_time + (h * 900) + 120,
                fee=0.0005,
                is_mixer_tx=is_mix
            )
            self.transactions[tx.tx_hash] = tx
            curr_b = next_b

        vasp_b_hw = self._get_vasp_wallet(vasp_b, chain, is_hot=True)
        tx_b_final = Transaction(
            tx_hash=self._generate_tx_hash(chain),
            chain=chain,
            from_address=curr_b,
            to_address=vasp_b_hw,
            amount=74.5,
            token_symbol="BTC",
            timestamp=base_time + (hops * 900) + 120,
            fee=0.0005
        )
        self.transactions[tx_b_final.tx_hash] = tx_b_final

    def _make_mixer_detour_path(
        self,
        source_wallet: str,
        wrong_vasp: str,
        correct_vasp: str,
        chain: Chain = Chain.ETH,
        tag_prefix: str = "detour"
    ) -> None:
        """
        Creates a misleading graph topology where:
        - Suspect reaches wrong_vasp in 2 hops via a mixer detour.
        - Suspect reaches correct_vasp in 4 hops via clean mules.
        Naive graph distance picks wrong_vasp (2 hops < 4 hops).
        """
        base_time = 1711300000

        # Short 2-hop path to wrong_vasp via mixer
        mixer = self._generate_address(chain, tag=f"{tag_prefix}_mixer_pool")
        self._register_benchmark_wallet(mixer, chain, "Obfuscation Mixer Detour", risk=0.98, is_mixer=True)
        wrong_hw = self._get_vasp_wallet(wrong_vasp, chain, is_hot=True)

        t_wrong_1 = Transaction(
            tx_hash=self._generate_tx_hash(chain),
            chain=chain,
            from_address=source_wallet,
            to_address=mixer,
            amount=60.0,
            token_symbol="ETH",
            timestamp=base_time,
            fee=0.004,
            is_mixer_tx=True
        )
        t_wrong_2 = Transaction(
            tx_hash=self._generate_tx_hash(chain),
            chain=chain,
            from_address=mixer,
            to_address=wrong_hw,
            amount=59.5,
            token_symbol="ETH",
            timestamp=base_time + 1800,
            fee=0.004,
            is_mixer_tx=True
        )
        self.transactions[t_wrong_1.tx_hash] = t_wrong_1
        self.transactions[t_wrong_2.tx_hash] = t_wrong_2

        # Longer 4-hop clean path to correct_vasp
        curr = source_wallet
        for h in range(3):
            mule = self._generate_address(chain, tag=f"{tag_prefix}_clean_mule_{h}")
            self._register_benchmark_wallet(mule, chain, f"Clean Mule Layer {h+1}", risk=0.3)
            tx = Transaction(
                tx_hash=self._generate_tx_hash(chain),
                chain=chain,
                from_address=curr,
                to_address=mule,
                amount=15.0 - (h * 0.2),
                token_symbol="ETH",
                timestamp=base_time + (h * 2400),
                fee=0.002
            )
            self.transactions[tx.tx_hash] = tx
            curr = mule

        correct_hw = self._get_vasp_wallet(correct_vasp, chain, is_hot=True)
        t_correct_final = Transaction(
            tx_hash=self._generate_tx_hash(chain),
            chain=chain,
            from_address=curr,
            to_address=correct_hw,
            amount=14.2,
            token_symbol="ETH",
            timestamp=base_time + 9600,
            fee=0.002
        )
        self.transactions[t_correct_final.tx_hash] = t_correct_final

    # -------------------------------------------------------------------------
    # Main Generation Pipeline
    # -------------------------------------------------------------------------

    def generate(self) -> Tuple[List[VASP], List[Wallet], List[Transaction], List[GroundTruthTestCase]]:
        """Main generation pipeline creating 9 VASPs, 550+ wallets, 2200+ transactions, 12 test cases."""
        self.wallets.clear()
        self.transactions.clear()
        self.vasps.clear()
        self.test_cases.clear()
        self.benchmark_addresses.clear()

        # 1. Initialize Seed VASPs with hot wallets and deposit sweepers
        for seed_data in SEED_VASPS:
            vasp_id = seed_data["id"]
            hot_wallets = []
            deposit_sweepers = []

            for ch in [Chain.BTC, Chain.ETH, Chain.TRON_TRC20]:
                hw_addr = self._generate_address(ch, tag=f"{vasp_id}_hot_{ch.value}")
                hot_wallets.append(hw_addr)
                self.wallets[hw_addr] = Wallet(
                    address=hw_addr,
                    chain=ch,
                    label=f"{seed_data['name']} Hot Wallet ({ch.value})",
                    vasp_id=vasp_id,
                    is_vasp_hot_wallet=True,
                    is_vasp_deposit_sweeper=False,
                    risk_score=0.05
                )

                sw_addr = self._generate_address(ch, tag=f"{vasp_id}_sweeper_{ch.value}")
                deposit_sweepers.append(sw_addr)
                self.wallets[sw_addr] = Wallet(
                    address=sw_addr,
                    chain=ch,
                    label=f"{seed_data['name']} Deposit Sweeper ({ch.value})",
                    vasp_id=vasp_id,
                    is_vasp_hot_wallet=False,
                    is_vasp_deposit_sweeper=True,
                    risk_score=0.1
                )

            vasp_obj = VASP(
                id=vasp_id,
                name=seed_data["name"],
                legal_name=seed_data.get("legal_name"),
                category=seed_data["category"],
                jurisdiction=seed_data["jurisdiction"],
                fiu_ind_registered=seed_data["fiu_ind_registered"],
                fiu_registration_number=seed_data.get("fiu_registration_number"),
                hot_wallets=hot_wallets,
                deposit_sweepers=deposit_sweepers
            )
            self.vasps[vasp_id] = vasp_obj

        # 2. Build 8 Base Benchmark Test Cases
        self._build_benchmark_case_1_peeling_chain()
        self._build_benchmark_case_2_mixer_coinjoin()
        self._build_benchmark_case_3_nested_hops()
        self._build_benchmark_case_4_deposit_sweeper()
        self._build_benchmark_case_5_trc20_peeling()
        self._build_benchmark_case_6_eth_multi_hop_mixer()
        self._build_benchmark_case_7_btc_ransomware_nested()
        self._build_benchmark_case_8_overseas_drain()

        # 3. Build 4 Hardened Benchmark Test Cases (CASE-101..CASE-104)
        self._build_benchmark_case_101_cross_chain_bridge()
        self._build_benchmark_case_102_proximity_tie_decoy()
        self._build_benchmark_case_103_mixer_detour()
        self._build_benchmark_case_104_cross_chain_tie()

        # 4. Generate background network of normal & suspicious activity
        self._generate_background_traffic(target_wallets=550, target_txs=2200)

        return (
            list(self.vasps.values()),
            list(self.wallets.values()),
            list(self.transactions.values()),
            self.test_cases
        )

    # -------------------------------------------------------------------------
    # Benchmark Case Builders (Existing 8 + 4 New Hardened)
    # -------------------------------------------------------------------------

    def _build_benchmark_case_1_peeling_chain(self):
        """Case 1: Bitcoin peeling chain terminating at CoinDCX hot wallet (3 hops). Baseline: PASS."""
        chain = Chain.BTC
        target_vasp = "coindcx"
        coindcx_hw = self._get_vasp_wallet(target_vasp, chain, is_hot=True)

        suspect = self._generate_address(chain, tag="suspect_case_1")
        hop1 = self._generate_address(chain, tag="peel_c1_h1")
        hop2 = self._generate_address(chain, tag="peel_c1_h2")

        for addr, lbl in [(suspect, "Suspect Wallet 1 (BTC Extortion)"), (hop1, "Peeling Hop 1"), (hop2, "Peeling Hop 2")]:
            self._register_benchmark_wallet(addr, chain, lbl, risk=0.85)

        base_time = 1710000000
        tx1 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=hop1, amount=12.5, token_symbol="BTC", timestamp=base_time, fee=0.0004, is_peeling_chain=True)
        tx2 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=hop1, to_address=hop2, amount=11.2, token_symbol="BTC", timestamp=base_time + 1800, fee=0.0004, is_peeling_chain=True)
        tx3 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=hop2, to_address=coindcx_hw, amount=10.0, token_symbol="BTC", timestamp=base_time + 3600, fee=0.0004, is_peeling_chain=True)

        for tx in [tx1, tx2, tx3]:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-001",
            case_description="BTC Peeling chain funneling ransomware loot to CoinDCX",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=3,
            expected_min_hops=3,
            expected_max_hops=3,
            expected_confidence_range=(0.80, 0.95),
            laundering_pattern=LaunderingPattern.PEELING.value,
            chain=chain,
            is_cross_chain=False,
            naive_proximity_will_fail=False
        ))

    def _build_benchmark_case_2_mixer_coinjoin(self):
        """Case 2: ETH mixer pass-through with decoy to Bybit at 2 hops. Baseline: FAIL (picks Bybit)."""
        chain = Chain.ETH
        target_vasp = "wazirx"
        decoy_vasp = "bybit"
        wazirx_sw = self._get_vasp_wallet(target_vasp, chain, is_hot=False)
        bybit_hw = self._get_vasp_wallet(decoy_vasp, chain, is_hot=True)

        suspect = self._generate_address(chain, tag="suspect_case_2")
        mixer_pool = self._generate_address(chain, tag="tornado_mock_pool")
        collector = self._generate_address(chain, tag="mixer_collector_c2")

        self._register_benchmark_wallet(suspect, chain, "Suspect Wallet 2 (Phishing Syndicate)", risk=0.92)
        self._register_benchmark_wallet(mixer_pool, chain, "Obfuscation Pool / Mixer", risk=0.98, is_mixer=True)
        self._register_benchmark_wallet(collector, chain, "Consolidation Wallet", risk=0.75)

        base_time = 1710100000
        # Target path: Suspect -> mixer -> collector -> WazirX (3 hops)
        tx1 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=mixer_pool, amount=50.0, token_symbol="ETH", timestamp=base_time, fee=0.005, is_mixer_tx=True)
        tx2 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=mixer_pool, to_address=collector, amount=49.2, token_symbol="ETH", timestamp=base_time + 7200, fee=0.004, is_mixer_tx=True)
        tx3 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=collector, to_address=wazirx_sw, amount=48.5, token_symbol="ETH", timestamp=base_time + 10800, fee=0.003)

        # Misleading decoy edge: mixer -> Bybit (making Bybit reachable in only 2 hops!)
        tx_decoy = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=mixer_pool, to_address=bybit_hw, amount=120.0, token_symbol="ETH", timestamp=base_time + 3600, fee=0.005, is_mixer_tx=True)

        for tx in [tx1, tx2, tx3, tx_decoy]:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-002",
            case_description="ETH mixer pass-through laundering phishing proceeds to WazirX (decoy Bybit closer)",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=3,
            expected_min_hops=2,
            expected_max_hops=3,
            expected_confidence_range=(0.75, 0.90),
            laundering_pattern=LaunderingPattern.MIXER.value,
            chain=chain,
            requires_behavioral_fingerprint=True,
            is_cross_chain=False,
            naive_proximity_will_fail=True
        ))

    def _build_benchmark_case_3_nested_hops(self):
        """Case 3: TRON USDT -> ETH Cross-chain bridge to ZebPay with 2-hop decoy to Bybit. Baseline: FAIL."""
        target_vasp = "zebpay"
        decoy_vasp = "bybit"
        zebpay_hw = self._get_vasp_wallet(target_vasp, Chain.ETH, is_hot=True)
        bybit_hw_tron = self._get_vasp_wallet(decoy_vasp, Chain.TRON_TRC20, is_hot=True)

        suspect = self._generate_address(Chain.TRON_TRC20, tag="suspect_case_3")
        layer1 = self._generate_address(Chain.TRON_TRC20, tag="nested_layer_1")
        bridge_eth = self._generate_address(Chain.ETH, tag="bridge_eth_c3")
        layer3 = self._generate_address(Chain.ETH, tag="transit_eth_c3")

        self._register_benchmark_wallet(suspect, Chain.TRON_TRC20, "Suspect Wallet 3 (Loan Fraud)", risk=0.78)
        self._register_benchmark_wallet(layer1, Chain.TRON_TRC20, "Mule Layer 1 (TRON)", risk=0.75)
        self._register_benchmark_wallet(bridge_eth, Chain.ETH, "Cross-Chain Swap Output (ETH)", risk=0.3)
        self._register_benchmark_wallet(layer3, Chain.ETH, "Transit Aggregator (ETH)", risk=0.5)

        base_time = 1710200000
        # Cross-chain path: TRON -> TRON -> ETH -> ETH -> ZebPay ETH HW (4 hops)
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(Chain.TRON_TRC20), chain=Chain.TRON_TRC20, from_address=suspect, to_address=layer1, amount=75000.0, token_symbol="USDT", timestamp=base_time, fee=1.5),
            Transaction(tx_hash=self._generate_tx_hash(Chain.ETH), chain=Chain.ETH, from_address=layer1, to_address=bridge_eth, amount=24.5, token_symbol="ETH", timestamp=base_time + 1200, fee=0.003),
            Transaction(tx_hash=self._generate_tx_hash(Chain.ETH), chain=Chain.ETH, from_address=bridge_eth, to_address=layer3, amount=24.4, token_symbol="ETH", timestamp=base_time + 2400, fee=0.003),
            Transaction(tx_hash=self._generate_tx_hash(Chain.ETH), chain=Chain.ETH, from_address=layer3, to_address=zebpay_hw, amount=24.3, token_symbol="ETH", timestamp=base_time + 3600, fee=0.003),
            # Decoy: layer1 -> Bybit TRON HW (2 hops from suspect!)
            Transaction(tx_hash=self._generate_tx_hash(Chain.TRON_TRC20), chain=Chain.TRON_TRC20, from_address=layer1, to_address=bybit_hw_tron, amount=150000.0, token_symbol="USDT", timestamp=base_time + 600, fee=1.5)
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-003",
            case_description="Cross-chain TRC-20 USDT into ETH layering terminating at ZebPay (decoy at 2 hops)",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=4,
            expected_min_hops=2,
            expected_max_hops=4,
            expected_confidence_range=(0.70, 0.88),
            laundering_pattern=LaunderingPattern.NESTED.value,
            chain=Chain.TRON_TRC20,
            is_cross_chain=True,
            requires_behavioral_fingerprint=True,
            naive_proximity_will_fail=True
        ))

    def _build_benchmark_case_4_deposit_sweeper(self):
        """Case 4: Fast deposit sweeper flow terminating at CoinSwitch (1 hop). Baseline: PASS."""
        chain = Chain.ETH
        target_vasp = "coinswitch"
        coinswitch_hw = self._get_vasp_wallet(target_vasp, chain, is_hot=True)
        coinswitch_sw = self._get_vasp_wallet(target_vasp, chain, is_hot=False)

        suspect = self._generate_address(chain, tag="suspect_case_4")
        self._register_benchmark_wallet(suspect, chain, "Suspect Wallet 4 (Ponzi Scheme)", risk=0.88)

        base_time = 1710300000
        tx1 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=coinswitch_sw, amount=22.0, token_symbol="ETH", timestamp=base_time, fee=0.002, is_sweeper_tx=True)
        tx2 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=coinswitch_sw, to_address=coinswitch_hw, amount=21.99, token_symbol="ETH", timestamp=base_time + 300, fee=0.001, is_sweeper_tx=True)

        for tx in [tx1, tx2]:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-004",
            case_description="ETH sweeper consolidation into CoinSwitch primary hot wallet",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=1,
            expected_min_hops=1,
            expected_max_hops=2,
            expected_confidence_range=(0.88, 0.98),
            laundering_pattern=LaunderingPattern.SWEEPER.value,
            chain=chain,
            is_cross_chain=False,
            naive_proximity_will_fail=False
        ))

    def _build_benchmark_case_5_trc20_peeling(self):
        """Case 5: High-speed TRC-20 peeling chain reaching Mudrex (3 hops). Baseline: PASS."""
        chain = Chain.TRON_TRC20
        target_vasp = "mudrex"
        mudrex_hw = self._get_vasp_wallet(target_vasp, chain, is_hot=True)

        suspect = self._generate_address(chain, tag="suspect_case_5")
        h1 = self._generate_address(chain, tag="trc20_peel_1")
        h2 = self._generate_address(chain, tag="trc20_peel_2")

        for addr, lbl in [(suspect, "Suspect Wallet 5 (Illegal Gaming App)"), (h1, "TRC-20 Peel Hop 1"), (h2, "TRC-20 Peel Hop 2")]:
            self._register_benchmark_wallet(addr, chain, lbl, risk=0.82)

        base_time = 1710400000
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=h1, amount=150000.0, token_symbol="USDT", timestamp=base_time, fee=1.5, is_peeling_chain=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=h1, to_address=h2, amount=135000.0, token_symbol="USDT", timestamp=base_time + 600, fee=1.5, is_peeling_chain=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=h2, to_address=mudrex_hw, amount=120000.0, token_symbol="USDT", timestamp=base_time + 1200, fee=1.5, is_peeling_chain=True),
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-005",
            case_description="High-value TRC-20 USDT gaming syndicate peeling change to Mudrex",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=3,
            expected_min_hops=3,
            expected_max_hops=3,
            expected_confidence_range=(0.78, 0.92),
            laundering_pattern=LaunderingPattern.PEELING.value,
            chain=chain,
            is_cross_chain=False,
            naive_proximity_will_fail=False
        ))

    def _build_benchmark_case_6_eth_multi_hop_mixer(self):
        """Case 6: ETH multi-hop mixer fan-out to Giottus (4 hops) with 2-hop decoy to Bybit. Baseline: FAIL."""
        chain = Chain.ETH
        target_vasp = "giottus"
        decoy_vasp = "bybit"
        giottus_hw = self._get_vasp_wallet(target_vasp, chain, is_hot=True)
        bybit_hw = self._get_vasp_wallet(decoy_vasp, chain, is_hot=True)

        suspect = self._generate_address(chain, tag="suspect_case_6")
        split1 = self._generate_address(chain, tag="mixer_split_1")
        split2 = self._generate_address(chain, tag="mixer_split_2")
        agg = self._generate_address(chain, tag="mixer_agg")

        for addr, lbl in [(suspect, "Suspect Wallet 6 (Darknet Market Vendor)"), (split1, "Mixer Intermediate A"), (split2, "Mixer Intermediate B"), (agg, "Relay Sweeper")]:
            self._register_benchmark_wallet(addr, chain, lbl, risk=0.90, is_mixer=(addr == split1))

        base_time = 1710500000
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=split1, amount=30.0, token_symbol="ETH", timestamp=base_time, fee=0.003, is_mixer_tx=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=split1, to_address=split2, amount=29.8, token_symbol="ETH", timestamp=base_time + 1500, fee=0.003, is_mixer_tx=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=split2, to_address=agg, amount=29.6, token_symbol="ETH", timestamp=base_time + 3000, fee=0.003),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=agg, to_address=giottus_hw, amount=29.4, token_symbol="ETH", timestamp=base_time + 4500, fee=0.003),
            # Decoy: split1 -> Bybit HW (2 hops from suspect!)
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=split1, to_address=bybit_hw, amount=80.0, token_symbol="ETH", timestamp=base_time + 1000, fee=0.004, is_mixer_tx=True)
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-006",
            case_description="Darknet market revenue laundered across multi-hop relay to Giottus (decoy closer)",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=4,
            expected_min_hops=2,
            expected_max_hops=4,
            expected_confidence_range=(0.72, 0.86),
            laundering_pattern=LaunderingPattern.NESTED.value,
            chain=chain,
            requires_behavioral_fingerprint=True,
            is_cross_chain=False,
            naive_proximity_will_fail=True
        ))

    def _build_benchmark_case_7_btc_ransomware_nested(self):
        """Case 7: Bitcoin complex ransomware peeling ending at Unocoin (3 hops). Baseline: PASS."""
        chain = Chain.BTC
        target_vasp = "unocoin"
        unocoin_hw = self._get_vasp_wallet(target_vasp, chain, is_hot=True)

        suspect = self._generate_address(chain, tag="suspect_case_7")
        h1 = self._generate_address(chain, tag="btc_ext_1")
        h2 = self._generate_address(chain, tag="btc_ext_2")

        for addr, lbl in [(suspect, "Suspect Wallet 7 (Hospital Ransomware Group)"), (h1, "Ransomware Mule 1"), (h2, "Ransomware Mule 2")]:
            self._register_benchmark_wallet(addr, chain, lbl, risk=0.96)

        base_time = 1710600000
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=h1, amount=8.4, token_symbol="BTC", timestamp=base_time, fee=0.0005, is_peeling_chain=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=h1, to_address=h2, amount=8.2, token_symbol="BTC", timestamp=base_time + 2000, fee=0.0005, is_peeling_chain=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=h2, to_address=unocoin_hw, amount=8.0, token_symbol="BTC", timestamp=base_time + 4000, fee=0.0005, is_peeling_chain=True),
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-007",
            case_description="Critical infrastructure ransomware extortion funneled to Unocoin",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=3,
            expected_min_hops=3,
            expected_max_hops=3,
            expected_confidence_range=(0.79, 0.93),
            laundering_pattern=LaunderingPattern.PEELING.value,
            chain=chain,
            is_cross_chain=False,
            naive_proximity_will_fail=False
        ))

    def _build_benchmark_case_8_overseas_drain(self):
        """Case 8: BTC -> TRON cross-chain capital flight draining to offshore Binance (3 hops). Baseline: PASS."""
        target_vasp = "binance_offshore"
        binance_hw = self._get_vasp_wallet(target_vasp, Chain.TRON_TRC20, is_hot=True)

        suspect = self._generate_address(Chain.BTC, tag="suspect_case_8")
        bridge_btc = self._generate_address(Chain.BTC, tag="hawala_bridge_btc")
        courier_tron = self._generate_address(Chain.TRON_TRC20, tag="hawala_courier_tron")

        self._register_benchmark_wallet(suspect, Chain.BTC, "Suspect Wallet 8 (Hawala Capital Flight)", risk=0.89)
        self._register_benchmark_wallet(bridge_btc, Chain.BTC, "Cross-Chain Hawala Bridge (BTC)", risk=0.7)
        self._register_benchmark_wallet(courier_tron, Chain.TRON_TRC20, "Offshore Transit Courier (TRON)", risk=0.8)

        base_time = 1710700000
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(Chain.BTC), chain=Chain.BTC, from_address=suspect, to_address=bridge_btc, amount=12.5, token_symbol="BTC", timestamp=base_time, fee=0.0005),
            Transaction(tx_hash=self._generate_tx_hash(Chain.TRON_TRC20), chain=Chain.TRON_TRC20, from_address=bridge_btc, to_address=courier_tron, amount=500000.0, token_symbol="USDT", timestamp=base_time + 600, fee=2.0),
            Transaction(tx_hash=self._generate_tx_hash(Chain.TRON_TRC20), chain=Chain.TRON_TRC20, from_address=courier_tron, to_address=binance_hw, amount=499500.0, token_symbol="USDT", timestamp=base_time + 1200, fee=2.0),
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-008",
            case_description="Cross-chain BTC to TRC-20 Hawala evasion routing flight into offshore non-compliant exchange",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=3,
            expected_min_hops=3,
            expected_max_hops=3,
            expected_confidence_range=(0.85, 0.96),
            laundering_pattern=LaunderingPattern.DIRECT.value,
            chain=Chain.BTC,
            is_cross_chain=True,
            naive_proximity_will_fail=False
        ))

    # -------------------------------------------------------------------------
    # 4 Hardened Cases (CASE-101..CASE-104)
    # -------------------------------------------------------------------------

    def _build_benchmark_case_101_cross_chain_bridge(self):
        """
        CASE-101: Cross-chain bridge hop: BTC -> ETH -> TRON_TRC20 terminating at CoinDCX.
        5 hops total across 3 chains. Baseline: PASS.
        """
        suspect = self._generate_address(Chain.BTC, tag="suspect_case_101")
        self._register_benchmark_wallet(suspect, Chain.BTC, "Suspect Wallet 101 (Multi-Chain Cyber Syndicate)", risk=0.91)
        target_vasp = "coindcx"

        # BTC -> ETH -> TRON corridor
        self._make_cross_chain_bridge_path(
            source_wallet=suspect,
            target_vasp=target_vasp,
            chains=[Chain.BTC, Chain.ETH, Chain.TRON_TRC20],
            tag_prefix="c101"
        )

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-101",
            case_description="Cross-chain bridge hop traversing BTC -> ETH -> TRON_TRC20 to CoinDCX",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=5,
            expected_min_hops=5,
            expected_max_hops=5,
            expected_confidence_range=(0.82, 0.94),
            laundering_pattern=LaunderingPattern.NESTED.value,
            chain=Chain.BTC,
            is_cross_chain=True,
            has_proximity_tie=False,
            naive_proximity_will_fail=False
        ))

    def _build_benchmark_case_102_proximity_tie_decoy(self):
        """
        CASE-102: Proximity tie with decoy.
        Two VASPs reachable at EXACT same hop count (3 hops):
        - Expected: ZebPay (clean path)
        - Decoy: WazirX (mixer path with higher volume, ranks #1 in pure proximity). Baseline: FAIL.
        """
        suspect = self._generate_address(Chain.BTC, tag="suspect_case_102")
        self._register_benchmark_wallet(suspect, Chain.BTC, "Suspect Wallet 102 (Equal-Distance Multi-Launderer)", risk=0.86)
        expected_vasp = "zebpay"
        decoy_vasp = "wazirx"

        self._make_proximity_tie_path(
            source_wallet=suspect,
            vasp_a=expected_vasp,
            vasp_b=decoy_vasp,
            hops=3,
            chain=Chain.BTC,
            tag_prefix="c102"
        )

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-102",
            case_description="Proximity tie with decoy: ZebPay (clean) vs WazirX (mixer decoy) both at 3 hops",
            suspect_wallet=suspect,
            expected_vasp=expected_vasp,
            expected_graph_distance=3,
            expected_min_hops=3,
            expected_max_hops=3,
            expected_confidence_range=(0.75, 0.90),
            laundering_pattern=LaunderingPattern.MIXER.value,
            chain=Chain.BTC,
            is_cross_chain=False,
            has_proximity_tie=True,
            requires_behavioral_fingerprint=True,
            naive_proximity_will_fail=True
        ))

    def _build_benchmark_case_103_mixer_detour(self):
        """
        CASE-103: Mixer detour makes naive distance wrong.
        Suspect -> 2 hops via mixer to Bybit (WRONG VASP).
        Suspect -> 4 hops via clean chain to WazirX (CORRECT VASP).
        Naive distance picks Bybit (2 hops < 4 hops). Baseline: FAIL.
        """
        suspect = self._generate_address(Chain.ETH, tag="suspect_case_103")
        self._register_benchmark_wallet(suspect, Chain.ETH, "Suspect Wallet 103 (Deceptive Mixer Detour)", risk=0.94)
        wrong_vasp = "bybit"
        correct_vasp = "wazirx"

        self._make_mixer_detour_path(
            source_wallet=suspect,
            wrong_vasp=wrong_vasp,
            correct_vasp=correct_vasp,
            chain=Chain.ETH,
            tag_prefix="c103"
        )

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-103",
            case_description="Mixer detour makes naive distance wrong: 2-hop mixer path to Bybit vs 4-hop clean to WazirX",
            suspect_wallet=suspect,
            expected_vasp=correct_vasp,
            expected_graph_distance=4,
            expected_min_hops=2,
            expected_max_hops=4,
            expected_confidence_range=(0.78, 0.92),
            laundering_pattern=LaunderingPattern.MIXER.value,
            chain=Chain.ETH,
            is_cross_chain=False,
            has_proximity_tie=False,
            requires_behavioral_fingerprint=True,
            naive_proximity_will_fail=True
        ))

    def _build_benchmark_case_104_cross_chain_tie(self):
        """
        CASE-104: Cross-chain + tie + mixer all in one case.
        Suspect on ETH:
        - Decoy path on ETH at 2 hops via mixer to Binance Offshore.
        - Correct path to TRON at 3 hops via cross-chain bridge to Mudrex.
        Naive proximity picks Binance Offshore. Baseline: FAIL.
        """
        chain_src = Chain.ETH
        chain_dst = Chain.TRON_TRC20
        suspect = self._generate_address(chain_src, tag="suspect_case_104")
        self._register_benchmark_wallet(suspect, chain_src, "Suspect Wallet 104 (Cross-Chain Mixer Syndicate)", risk=0.95)

        decoy_vasp = "binance_offshore"
        expected_vasp = "mudrex"

        base_time = 1711400000

        # Decoy path on ETH via mixer: suspect -> mixer -> Binance Offshore ETH HW (2 hops)
        mixer = self._generate_address(chain_src, tag="c104_mixer")
        self._register_benchmark_wallet(mixer, chain_src, "ETH Tornado Mixer Pool", risk=0.98, is_mixer=True)
        binance_hw_eth = self._get_vasp_wallet(decoy_vasp, chain_src, is_hot=True)

        t_decoy_1 = Transaction(
            tx_hash=self._generate_tx_hash(chain_src),
            chain=chain_src,
            from_address=suspect,
            to_address=mixer,
            amount=40.0,
            token_symbol="ETH",
            timestamp=base_time,
            fee=0.005,
            is_mixer_tx=True
        )
        t_decoy_2 = Transaction(
            tx_hash=self._generate_tx_hash(chain_src),
            chain=chain_src,
            from_address=mixer,
            to_address=binance_hw_eth,
            amount=39.5,
            token_symbol="ETH",
            timestamp=base_time + 1200,
            fee=0.005,
            is_mixer_tx=True
        )
        self.transactions[t_decoy_1.tx_hash] = t_decoy_1
        self.transactions[t_decoy_2.tx_hash] = t_decoy_2

        # Expected cross-chain path: suspect -> bridge_eth -> mule_tron -> Mudrex TRON HW (3 hops)
        bridge_dep = self._generate_address(chain_src, tag="c104_bridge_dep")
        mule_tron = self._generate_address(chain_dst, tag="c104_mule_tron")
        self._register_benchmark_wallet(bridge_dep, chain_src, "Cross-Chain Portal Deposit (ETH)", risk=0.3)
        self._register_benchmark_wallet(mule_tron, chain_dst, "Cross-Chain Mule Payout (TRON)", risk=0.6)

        mudrex_hw_tron = self._get_vasp_wallet(expected_vasp, chain_dst, is_hot=True)

        t_exp_1 = Transaction(
            tx_hash=self._generate_tx_hash(chain_src),
            chain=chain_src,
            from_address=suspect,
            to_address=bridge_dep,
            amount=15.0,
            token_symbol="ETH",
            timestamp=base_time,
            fee=0.002
        )
        t_exp_2 = Transaction(
            tx_hash=self._generate_tx_hash(chain_dst),
            chain=chain_dst,
            from_address=bridge_dep,
            to_address=mule_tron,
            amount=50000.0,
            token_symbol="USDT",
            timestamp=base_time + 1800,
            fee=1.5
        )
        t_exp_3 = Transaction(
            tx_hash=self._generate_tx_hash(chain_dst),
            chain=chain_dst,
            from_address=mule_tron,
            to_address=mudrex_hw_tron,
            amount=49800.0,
            token_symbol="USDT",
            timestamp=base_time + 3600,
            fee=1.5
        )
        for tx in [t_exp_1, t_exp_2, t_exp_3]:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-104",
            case_description="Cross-chain + tie: 2-hop mixer decoy to Binance vs 3-hop cross-chain bridge to Mudrex",
            suspect_wallet=suspect,
            expected_vasp=expected_vasp,
            expected_graph_distance=3,
            expected_min_hops=2,
            expected_max_hops=3,
            expected_confidence_range=(0.76, 0.90),
            laundering_pattern=LaunderingPattern.MIXER.value,
            chain=chain_src,
            is_cross_chain=True,
            has_proximity_tie=True,
            requires_behavioral_fingerprint=True,
            naive_proximity_will_fail=True
        ))

    # -------------------------------------------------------------------------
    # Background Traffic Generation (500+ Wallets, 2000+ Transactions)
    # -------------------------------------------------------------------------

    def _generate_background_traffic(self, target_wallets: int = 550, target_txs: int = 2200):
        """Creates hundreds of realistic wallets and interconnecting transactions."""
        chains = [Chain.BTC, Chain.ETH, Chain.TRON_TRC20]
        
        # 1. Create background trader/user wallets
        current_wallet_count = len(self.wallets)
        wallets_needed = max(0, target_wallets - current_wallet_count)

        bg_wallets_by_chain: Dict[Chain, List[str]] = {c: [] for c in chains}

        for i in range(wallets_needed):
            c = self.rng.choice(chains)
            addr = self._generate_address(c, tag=f"bg_user_{i}")
            risk = round(self.rng.betavariate(1.5, 5.0), 3)
            w = Wallet(
                address=addr,
                chain=c,
                label=f"Trader {i}" if risk < 0.3 else f"Unlabeled {i}",
                risk_score=risk
            )
            self.wallets[addr] = w
            bg_wallets_by_chain[c].append(addr)

        # Collect VASP deposit targets for organic exchange deposits
        vasp_targets_by_chain: Dict[Chain, List[str]] = {c: [] for c in chains}
        for vasp in self.vasps.values():
            for hw in vasp.hot_wallets:
                w = self.wallets.get(hw)
                if w:
                    vasp_targets_by_chain[w.chain].append(hw)
            for sw in vasp.deposit_sweepers:
                w = self.wallets.get(sw)
                if w:
                    vasp_targets_by_chain[w.chain].append(sw)

        # 2. Generate realistic background transactions across each chain
        current_tx_count = len(self.transactions)
        txs_needed = max(0, target_txs - current_tx_count)

        base_time = 1710000000
        token_map = {Chain.BTC: "BTC", Chain.ETH: "ETH", Chain.TRON_TRC20: "USDT"}

        for i in range(txs_needed):
            c = self.rng.choice(chains)
            senders = bg_wallets_by_chain[c]
            if len(senders) < 2:
                continue

            src = self.rng.choice(senders)
            
            # 80% P2P trader transfer, 20% organic deposit into a VASP
            if self.rng.random() < 0.20 and vasp_targets_by_chain[c]:
                dst = self.rng.choice(vasp_targets_by_chain[c])
            else:
                dst = self.rng.choice(senders)

            if src == dst:
                continue

            tok = token_map[c]
            if c == Chain.BTC:
                amt = round(self.rng.lognormvariate(-1.5, 1.2), 4)
                fee = round(self.rng.uniform(0.0001, 0.0008), 5)
            elif c == Chain.ETH:
                amt = round(self.rng.lognormvariate(0.5, 1.4), 3)
                fee = round(self.rng.uniform(0.001, 0.006), 4)
            else:  # TRON_TRC20
                amt = round(self.rng.uniform(50, 5000), 1)
                fee = 1.5

            timestamp = base_time + self.rng.randint(0, 86400 * 14)
            tx_h = self._generate_tx_hash(c)
            tx = Transaction(
                tx_hash=tx_h,
                chain=c,
                from_address=src,
                to_address=dst,
                amount=max(0.0001, amt),
                token_symbol=tok,
                timestamp=timestamp,
                fee=fee
            )
            self.transactions[tx_h] = tx

    def populate_store(self, store: GraphStore) -> None:
        """Pushes all generated entities into any GraphStore implementation."""
        for vasp in self.vasps.values():
            store.add_vasp(vasp)
        for wallet in self.wallets.values():
            store.add_wallet(wallet)
        for tx in self.transactions.values():
            store.add_transaction(tx)

    def export_json(self, file_path: str) -> None:
        """Exports full dataset to JSON for standalone offline inspections."""
        data = {
            "seed": self.seed,
            "vasps": [v.model_dump() for v in self.vasps.values()],
            "wallets": [w.model_dump() for w in self.wallets.values()],
            "transactions": [t.model_dump() for t in self.transactions.values()],
            "test_cases": [tc.model_dump() for tc in self.test_cases]
        }
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def generate_synthetic_data(seed: int = 42, store: Optional[GraphStore] = None) -> Tuple[List[VASP], List[Wallet], List[Transaction], List[GroundTruthTestCase]]:
    """Helper entry point for deterministic synthetic graph generation."""
    generator = SyntheticDataGenerator(seed=seed)
    vasps, wallets, transactions, test_cases = generator.generate()
    if store is not None:
        generator.populate_store(store)
    return vasps, wallets, transactions, test_cases
