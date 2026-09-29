# Money Migration Atlas (SIH26182) — 3-Minute Live Demo Script

**Theme:** Automated Cross-Chain Asset Flight Attribution & Legal Disclosure Package Generation  
**Target Duration:** Exactly 3 Minutes (180 Seconds)  
**Tone:** Authoritative, mathematically grounded, forensically objective. Zero hype.

---

## Quick Reference / Demo Checklist
- [ ] Backend running on `http://localhost:8000` (`python -m uvicorn app.main:app --port 8000`)
- [ ] Frontend running on `http://localhost:5173` (`npm run dev`)
- [ ] Browser window open to Dashboard (`http://localhost:5173`)
- [ ] Target benchmark cases ready:
  - **Primary Case (Agree / Confirmed):** `CASE-002` (WazirX Multi-hop flight, 2 hops, ETH)
  - **Disagreement / Triage Case:** `CASE-104` (Binance vs Mudrex ambiguity)

---

## 3-Minute Presentation Walkthrough

### [00:00 – 00:30] The Law Enforcement Problem & Solution Core
- **Spoken:**
  > "Distinguished jury, when criminal proceeds migrate across blockchains through peel chains and intermediary mixers, Indian law enforcement faces a critical hurdle: **Which regulated Indian VASP holds custody of the destination wallet?** 
  > Under Section 91 of the CrPC, lawful disclosure directives and emergency freeze notices can only be served to compliant reporting entities registered with FIU-IND. 
  > Existing closed commercial tools rely on proprietary black boxes that don't satisfy rigorous evidentiary scrutiny in court. 
  > **Money Migration Atlas** is an explainable, multi-model forensic pipeline combining topological Dijkstra traversal, 20-feature XGBoost gradient boosting, and Graph Neural Network message passing into an unblended, auditable attribution engine."

- **Action:**
  - Show the Dashboard landing view with the 12 SIH Benchmark test cases and live system metrics.
  - Point to the Model Architecture summary badges: **XGBoost (Brier Score 0.074)** and **GraphSAGE (0.86 ROC-AUC)**.

---

### [00:30 – 01:15] Topological Exploration & Multi-Model Inference (CASE-002)
- **Spoken:**
  > "Let's inspect **CASE-002**, a real-world simulation of an Ethereum cyber fraud drainer. 
  > With one click, our engine executes a 6-hop breadth traversal, reconstructs the transaction ego-subgraph in Cytoscape, and extracts features for independent models."

- **Action:**
  - Click on **CASE-002** in the Quick Scenario Selector.
  - The live inference progress timeline animates:
    1. *Dijkstra Shortest Path Traversal*
    2. *XGBoost 20-Feature Inference*
    3. *GraphSAGE Structural Message Passing*
    4. *Multi-Model Consensus Agreement*
  - The Investigation View loads in under 500ms.

- **Spoken:**
  > "Notice how our UI preserves strict evidentiary discipline: **Topological proximity rank and predictive confidence are NEVER blended into a single composite metric.**
  > Here, Dijkstra finds WazirX at 2 hops. XGBoost evaluates temporal dynamics and flow volume, giving **0.86 confidence**. GraphSAGE analyses the 2-hop structural neighborhood motif, yielding **0.86 confidence**. 
  > Because both independent mathematical paradigms agree above our calibrated 0.60 threshold, the consensus engine marks the verdict as **CONFIRMED**."

---

### [01:15 – 02:00] Plain-Language Explanations & Quantitative Counterfactuals
- **Spoken:**
  > "A forensic investigator does not present raw tensor logits to a magistrate; they require plain-language prose with mathematical provenance.
  > Below the graph, our Agentic Explainer summarizes the transaction flight in human terms, supported by SHAP feature attributions."

- **Action:**
  - Scroll down to the **Plain-Language Summary** and the **Forensic Counterfactuals** panel.
  - Highlight the counterfactual cards with explicit model attribution:
    - *"If flow volume dropped below 5,000 USDT, XGBoost confidence would drop from 0.86 to 0.41 (recomputed)."*
    - *"If the path bypassed mixer hops, GNN confidence would remain at 0.86 while XGBoost increases to 0.91."*

- **Spoken:**
  > "Notice the provenance tag on each counterfactual. Every 'what-if' statement explicitly names the underlying model—whether XGBoost or GraphSAGE—and specifies whether it was mathematically recomputed or estimated via Shapley values."

---

### [02:00 – 02:35] Chain of Custody & Court-Ready Evidence Package
- **Spoken:**
  > "To ensure digital chain of custody, every analysis step generates a SHA-256 fingerprint with millisecond-accurate timestamps and monotonic sequence tracking.
  > We provide an automated export of a court-ready dossier formatted specifically for Section 91 CrPC disclosure."

- **Action:**
  - Show the **Chain of Custody Timeline** with 5 strictly monotonic steps:
    1. `INGEST_WALLET`
    2. `GRAPH_TRAVERSAL`
    3. `FEATURE_EXTRACTION`
    4. `ML_INFERENCE`
    5. `CONSENSUS_ARBITRATION`
  - Click **Download Dossier (Markdown)** or **Inspect Raw Evidence JSON**.
  - Show the prominent notice: *'Court-ready dossier (not a certification). Proximity distance and predictive confidence maintained as strictly unblended parameters.'*

---

### [02:35 – 03:00] SAHYOG Mock Gateway & Responsible Legal Handoff
- **Spoken:**
  > "Finally, our pipeline closes the investigative loop by integrating with the Ministry of Home Affairs SAHYOG portal architecture.
  > The system validates whether the identified candidate is registered with FIU-IND. 
  > For WazirX—a compliant Indian reporting entity—the investigator can immediately dispatch an automated Section 91 disclosure directive.
  > If the suspect had routed to an offshore exchange like Bybit, the system legally prohibits automated SAHYOG dispatch and mandates diplomatic MLAT routing."

- **Action:**
  - Click **Submit Disclosure Request** in the **Route to SAHYOG** panel.
  - Watch the live lifecycle progression transition from **1. QUEUED** to **2. ROUTED** to **3. ACKNOWLEDGED** within 3 seconds.
  - Highlight the deterministic Request ID (`sahyog-xxxx`) and the persistent **MOCK GATEWAY** disclaimer.

- **Closing (Spoken):**
  > "Money Migration Atlas: transparent, calibrated, unblended, and court-ready. Thank you, and we welcome your questions."

---

## Speaker Defense Notes for Q&A

### 1. "Why not just use Chainalysis, TRM Labs, or Elliptic?"
- **Answer (30s):**
  > "Commercial tools are proprietary black boxes. When a defense attorney cross-examines an investigator in an Indian Sessions Court on how an attribution score was computed, a closed commercial vendor cannot provide open mathematical reproducibility.
  > Money Migration Atlas is fully open-source, uses dual orthogonal models (XGBoost tabular + GraphSAGE message passing), publishes explicit SHAP feature weights, asserts strict unblended scores, and costs zero licensing fees for state police cyber cells."

### 2. "How do you know your model is calibrated and not overconfident?"
- **Answer (30s):**
  > "We calibrated our XGBoost classifier using isotonic probability calibration on out-of-fold validation splits. 
  > Our Brier score is **0.074** (where 0 is a perfect probabilistic forecast). In contrast to uncalibrated neural nets that cluster at 0.99 confidence even when guessing, our model outputs probabilities that reflect true empirical frequencies. Furthermore, under dual-model consensus, both models must independently score $\ge 0.60$ for a CONFIRMED verdict."

### 3. "What is your false positive rate, and what happens when the model is wrong?"
- **Answer (30s):**
  > "Across our 12-case ground-truth benchmark, our multi-model consensus engine produced **0 CONFIRMED-wrong attributions**. 
  > When the models disagree—such as in CASE-104 where topological proximity suggests an offshore exchange but tabular features suggest Mudrex—the system outputs **AMBIGUOUS** or **SINGLE_MODEL** and issues a clear human directive: *'Human Review Required: Do Not Issue Automated Freeze.'* The system fails gracefully into forensic triage rather than issuing false legal notices."

### 4. "Is this system ready for nationwide production deployment tomorrow?"
- **Answer (30s):**
  > "We maintain rigorous academic and engineering honesty regarding our current limitations:
  > First, our models were trained and benchmarked on synthetic graph generators replicating known on-chain laundering topologies; deployment on live Ethereum/Tron mainnet data requires retraining on real exchange-labeled clusters.
  > Second, our SAHYOG integration is a simulated mock router demonstrating API compliance with MHA specs.
  > Third, behavioral fingerprinting proved decorative on sparse synthetic transactions (Phase 6a audit). 
  > What is production-ready today is the underlying architecture: unblended scoring, dual-model consensus, explainability, and SHA-256 chain of custody."
