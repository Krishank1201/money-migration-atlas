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
            # P2WPKH Bech32 format
            return f"bc1q{h[:38]}"
        elif chain == Chain.ETH:
            # 0x prefixed 40 hex chars
            return f"0x{h[:40]}"
        elif chain == Chain.TRON_TRC20:
            # Base58-style starting with 'T'
            b58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            return "T" + "".join(self.rng.choices(b58, k=33))
        else:
            return f"0x{h[:40]}"

    def _generate_tx_hash(self, chain: Chain) -> str:
        raw = self._rand_hex(64)
        return f"0x{raw}" if chain in (Chain.ETH, Chain.POLYGON, Chain.BSC) else raw

    def generate(self) -> Tuple[List[VASP], List[Wallet], List[Transaction], List[GroundTruthTestCase]]:
        """Main generation pipeline creating 8+ VASPs, 500+ wallets, 2000+ transactions."""
        self.wallets.clear()
        self.transactions.clear()
        self.vasps.clear()
        self.test_cases.clear()

        # 1. Initialize Seed VASPs with hot wallets and deposit sweepers
        for seed_data in SEED_VASPS:
            vasp_id = seed_data["id"]
            hot_wallets = []
            deposit_sweepers = []

            # Create hot wallets across BTC, ETH, TRON
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

        # 2. Build 8 Benchmark Ground-Truth Test Cases with distinctive laundering topologies
        self._build_benchmark_case_1_peeling_chain()
        self._build_benchmark_case_2_mixer_coinjoin()
        self._build_benchmark_case_3_nested_hops()
        self._build_benchmark_case_4_deposit_sweeper()
        self._build_benchmark_case_5_trc20_peeling()
        self._build_benchmark_case_6_eth_multi_hop_mixer()
        self._build_benchmark_case_7_btc_ransomware_nested()
        self._build_benchmark_case_8_overseas_drain()

        # 3. Generate background network of normal & suspicious activity
        # To achieve 500+ wallets and 2000+ transactions
        self._generate_background_traffic(target_wallets=550, target_txs=2200)

        return (
            list(self.vasps.values()),
            list(self.wallets.values()),
            list(self.transactions.values()),
            self.test_cases
        )

    # -------------------------------------------------------------------------
    # Benchmark Case Builders
    # -------------------------------------------------------------------------

    def _build_benchmark_case_1_peeling_chain(self):
        """Case 1: Bitcoin peeling chain terminating at CoinDCX hot wallet (3 hops)."""
        chain = Chain.BTC
        target_vasp = "coindcx"
        coindcx_hw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_hot_wallet][0]

        suspect = self._generate_address(chain, tag="suspect_case_1")
        hop1 = self._generate_address(chain, tag="peel_c1_h1")
        hop2 = self._generate_address(chain, tag="peel_c1_h2")

        for addr, lbl in [(suspect, "Suspect Wallet 1 (BTC Extortion)"), (hop1, "Peeling Hop 1"), (hop2, "Peeling Hop 2")]:
            self.wallets[addr] = Wallet(address=addr, chain=chain, label=lbl, risk_score=0.85)

        base_time = 1710000000
        # Suspect -> Hop1 -> Hop2 -> CoinDCX Hot Wallet
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
            expected_confidence_range=(0.80, 0.95),
            laundering_pattern=LaunderingPattern.PEELING.value,
            chain=chain
        ))

    def _build_benchmark_case_2_mixer_coinjoin(self):
        """Case 2: Ethereum mixer fan-out/fan-in ending at WazirX deposit sweeper (4 hops)."""
        chain = Chain.ETH
        target_vasp = "wazirx"
        wazirx_sw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_deposit_sweeper][0]

        suspect = self._generate_address(chain, tag="suspect_case_2")
        mixer_pool = self._generate_address(chain, tag="tornado_mock_pool")
        collector = self._generate_address(chain, tag="mixer_collector_c2")

        self.wallets[suspect] = Wallet(address=suspect, chain=chain, label="Suspect Wallet 2 (Phishing Syndicate)", risk_score=0.92)
        self.wallets[mixer_pool] = Wallet(address=mixer_pool, chain=chain, label="Obfuscation Pool / Mixer", is_mixer=True, risk_score=0.98)
        self.wallets[collector] = Wallet(address=collector, chain=chain, label="Consolidation Wallet", risk_score=0.75)

        base_time = 1710100000
        tx1 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=mixer_pool, amount=50.0, token_symbol="ETH", timestamp=base_time, fee=0.005, is_mixer_tx=True)
        tx2 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=mixer_pool, to_address=collector, amount=49.2, token_symbol="ETH", timestamp=base_time + 7200, fee=0.004, is_mixer_tx=True)
        tx3 = Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=collector, to_address=wazirx_sw, amount=48.5, token_symbol="ETH", timestamp=base_time + 10800, fee=0.003, is_mixer_tx=False)

        for tx in [tx1, tx2, tx3]:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-002",
            case_description="ETH mixer pass-through laundering phishing proceeds to WazirX",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=3,
            expected_confidence_range=(0.75, 0.90),
            laundering_pattern=LaunderingPattern.MIXER.value,
            chain=chain
        ))

    def _build_benchmark_case_3_nested_hops(self):
        """Case 3: TRC-20 nested transit layering into ZebPay hot wallet (4 hops)."""
        chain = Chain.TRON_TRC20
        target_vasp = "zebpay"
        zebpay_hw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_hot_wallet][0]

        suspect = self._generate_address(chain, tag="suspect_case_3")
        layer1 = self._generate_address(chain, tag="nested_layer_1")
        layer2 = self._generate_address(chain, tag="nested_layer_2")
        layer3 = self._generate_address(chain, tag="nested_layer_3")

        for addr, lbl in [(suspect, "Suspect Wallet 3 (Loan Fraud)"), (layer1, "Mule Layer 1"), (layer2, "Mule Layer 2"), (layer3, "Transit Aggregator")]:
            self.wallets[addr] = Wallet(address=addr, chain=chain, label=lbl, risk_score=0.78)

        base_time = 1710200000
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=layer1, amount=75000.0, token_symbol="USDT", timestamp=base_time, fee=1.5),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=layer1, to_address=layer2, amount=74900.0, token_symbol="USDT", timestamp=base_time + 1200, fee=1.5),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=layer2, to_address=layer3, amount=74800.0, token_symbol="USDT", timestamp=base_time + 2400, fee=1.5),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=layer3, to_address=zebpay_hw, amount=74700.0, token_symbol="USDT", timestamp=base_time + 3600, fee=1.5),
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-003",
            case_description="TRC-20 USDT multi-hop layering into ZebPay exchange reserves",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=4,
            expected_confidence_range=(0.70, 0.88),
            laundering_pattern=LaunderingPattern.NESTED.value,
            chain=chain
        ))

    def _build_benchmark_case_4_deposit_sweeper(self):
        """Case 4: Fast deposit sweeper flow terminating at CoinSwitch (2 hops)."""
        chain = Chain.ETH
        target_vasp = "coinswitch"
        coinswitch_hw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_hot_wallet][0]
        coinswitch_sw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_deposit_sweeper][0]

        suspect = self._generate_address(chain, tag="suspect_case_4")
        self.wallets[suspect] = Wallet(address=suspect, chain=chain, label="Suspect Wallet 4 (Ponzi Scheme)", risk_score=0.88)

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
            expected_graph_distance=1,  # Direct hit to sweeper (1 hop) and 2 to HW
            expected_confidence_range=(0.88, 0.98),
            laundering_pattern=LaunderingPattern.SWEEPER.value,
            chain=chain
        ))

    def _build_benchmark_case_5_trc20_peeling(self):
        """Case 5: High-speed TRC-20 peeling chain reaching Mudrex (3 hops)."""
        chain = Chain.TRON_TRC20
        target_vasp = "mudrex"
        mudrex_hw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_hot_wallet][0]

        suspect = self._generate_address(chain, tag="suspect_case_5")
        h1 = self._generate_address(chain, tag="trc20_peel_1")
        h2 = self._generate_address(chain, tag="trc20_peel_2")

        for addr, lbl in [(suspect, "Suspect Wallet 5 (Illegal Gaming App)"), (h1, "TRC-20 Peel Hop 1"), (h2, "TRC-20 Peel Hop 2")]:
            self.wallets[addr] = Wallet(address=addr, chain=chain, label=lbl, risk_score=0.82)

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
            expected_confidence_range=(0.78, 0.92),
            laundering_pattern=LaunderingPattern.PEELING.value,
            chain=chain
        ))

    def _build_benchmark_case_6_eth_multi_hop_mixer(self):
        """Case 6: ETH multi-hop mixer fan-out ending at Giottus (4 hops)."""
        chain = Chain.ETH
        target_vasp = "giottus"
        giottus_hw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_hot_wallet][0]

        suspect = self._generate_address(chain, tag="suspect_case_6")
        split1 = self._generate_address(chain, tag="mixer_split_1")
        split2 = self._generate_address(chain, tag="mixer_split_2")
        agg = self._generate_address(chain, tag="mixer_agg")

        for addr, lbl in [(suspect, "Suspect Wallet 6 (Darknet Market Vendor)"), (split1, "Mixer Intermediate A"), (split2, "Mixer Intermediate B"), (agg, "Relay Sweeper")]:
            self.wallets[addr] = Wallet(address=addr, chain=chain, label=lbl, risk_score=0.90)

        base_time = 1710500000
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=split1, amount=30.0, token_symbol="ETH", timestamp=base_time, fee=0.003, is_mixer_tx=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=split1, to_address=split2, amount=29.8, token_symbol="ETH", timestamp=base_time + 1500, fee=0.003, is_mixer_tx=True),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=split2, to_address=agg, amount=29.6, token_symbol="ETH", timestamp=base_time + 3000, fee=0.003),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=agg, to_address=giottus_hw, amount=29.4, token_symbol="ETH", timestamp=base_time + 4500, fee=0.003),
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-006",
            case_description="Darknet market revenue laundered across multi-hop relay to Giottus",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=4,
            expected_confidence_range=(0.72, 0.86),
            laundering_pattern=LaunderingPattern.NESTED.value,
            chain=chain
        ))

    def _build_benchmark_case_7_btc_ransomware_nested(self):
        """Case 7: Bitcoin complex ransomware peeling ending at Unocoin (3 hops)."""
        chain = Chain.BTC
        target_vasp = "unocoin"
        unocoin_hw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_hot_wallet][0]

        suspect = self._generate_address(chain, tag="suspect_case_7")
        h1 = self._generate_address(chain, tag="btc_ext_1")
        h2 = self._generate_address(chain, tag="btc_ext_2")

        for addr, lbl in [(suspect, "Suspect Wallet 7 (Hospital Ransomware Group)"), (h1, "Ransomware Mule 1"), (h2, "Ransomware Mule 2")]:
            self.wallets[addr] = Wallet(address=addr, chain=chain, label=lbl, risk_score=0.96)

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
            expected_confidence_range=(0.79, 0.93),
            laundering_pattern=LaunderingPattern.PEELING.value,
            chain=chain
        ))

    def _build_benchmark_case_8_overseas_drain(self):
        """Case 8: Overseas capital flight draining to non-compliant offshore Binance (2 hops)."""
        chain = Chain.TRON_TRC20
        target_vasp = "binance_offshore"
        binance_hw = [w.address for w in self.wallets.values() if w.vasp_id == target_vasp and w.chain == chain and w.is_vasp_hot_wallet][0]

        suspect = self._generate_address(chain, tag="suspect_case_8")
        h1 = self._generate_address(chain, tag="overseas_courier")

        for addr, lbl in [(suspect, "Suspect Wallet 8 (Hawala Capital Flight)"), (h1, "Offshore Transit Courier")]:
            self.wallets[addr] = Wallet(address=addr, chain=chain, label=lbl, risk_score=0.89)

        base_time = 1710700000
        txs = [
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=suspect, to_address=h1, amount=500000.0, token_symbol="USDT", timestamp=base_time, fee=2.0),
            Transaction(tx_hash=self._generate_tx_hash(chain), chain=chain, from_address=h1, to_address=binance_hw, amount=499500.0, token_symbol="USDT", timestamp=base_time + 900, fee=2.0),
        ]
        for tx in txs:
            self.transactions[tx.tx_hash] = tx

        self.test_cases.append(GroundTruthTestCase(
            case_id="CASE-008",
            case_description="Hawala evasion routing massive USDT flight into offshore non-compliant exchange",
            suspect_wallet=suspect,
            expected_vasp=target_vasp,
            expected_graph_distance=2,
            expected_confidence_range=(0.85, 0.96),
            laundering_pattern=LaunderingPattern.DIRECT.value,
            chain=chain
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
            risk = round(self.rng.betavariate(1.5, 5.0), 3)  # skewed towards low risk
            w = Wallet(
                address=addr,
                chain=c,
                label=f"Trader {i}" if risk < 0.3 else f"Unlabeled {i}",
                risk_score=risk
            )
            self.wallets[addr] = w
            bg_wallets_by_chain[c].append(addr)

        # Collect VASP deposit targets for realistic organic exchange deposits
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

            timestamp = base_time + self.rng.randint(0, 86400 * 14)  # 2 weeks span
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
