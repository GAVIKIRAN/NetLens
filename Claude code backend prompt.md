# Claude Code Prompt — XAI-NIDS Backend

Paste everything below the line into Claude Code, in the project folder.

---

You are the lead machine-learning and backend engineer for my MCA final-year project. I need you
to build and verify the **complete backend** of an Explainable AI-based Network Intrusion
Detection System. The dashboard (Streamlit UI) is a separate concern — your job is every layer
beneath it, plus the interfaces the dashboard reads from.

## 0. Ground rules — do not violate these

1. **Write complete, runnable files.** No snippets, no fragments, no `# TODO`, no
   `# implement this later`, no `...`, no `pass  # add your code here`. Every function you
   write has a real body.
2. **Never invent a number.** Accuracy, precision, recall, F1, training time, flow counts —
   if you have not measured it by running code, do not write it down. Where a number belongs
   but has not been measured, write `[not yet measured — run scripts/evaluate_model.py]`.
3. **Never claim something works unless you ran it.** If you could not run it, say so plainly
   and say why.
4. **Defensive security only.** This system observes traffic and reports. Do not write, and do
   not suggest: exploit code, malware, credential harvesting, packet injection, traffic
   flooding or denial-of-service generation, or offensive scanning tools. For demonstration,
   build a **safe simulation mode** that synthesises flow records and replays stored captures.
   Nothing in this project ever transmits an attack.
5. **Capture is read-only.** Sniff, parse, and discard. Never inject, never modify, never
   forward.
6. **Privacy.** Store connection metadata only — addresses, ports, protocol, timings, counts,
   statistics. Never store packet payloads.

## 1. The design problem you must solve FIRST — do not skip this

The model is trained on **CIC-IDS2017**, whose features were produced offline by
**CICFlowMeter**. At detection time the features must come from **live packet capture**. These
two sources do not naturally agree, and a system that ignores the gap produces a model that
scores well in training and is meaningless in production.

**Before you write any model code**, do this and report back:

1. Enumerate every column in CIC-IDS2017.
2. For each column, decide whether it can be computed from live packets with the same
   definition and the same units. Classify each as **derivable**, **derivable with a documented
   approximation**, or **not derivable**.
3. Identify columns that must be excluded because they leak the label or are degenerate
   (constant, always zero, duplicated).
4. Write the surviving set into **one module that is the single source of truth**:
   `src/features/contract.py`. It must export the feature list, the exclusion list with a
   reason per exclusion, and helpers that align any dataframe to the contract order.
5. **Both** the training pipeline **and** the live inference path import their feature list from
   this module. Neither is allowed to define its own. Column order must be impossible to drift.
6. Write a short note explaining the strategy and its trade-offs.

Specific decisions I want you to make deliberately and justify:

- **`Destination Port`** is a label leak on CIC-IDS2017 (attacks were run against fixed ports,
  so the model learns the experiment rather than the attack). Default to excluding it. If you
  include it, bucket it (well-known / registered / dynamic) rather than using the raw integer,
  and make it a config flag that defaults to off.
- **Bulk-transfer columns** (`Fwd Avg Bytes/Bulk` and its seven relatives) are all zero across
  the public CSVs. Drop them and say so.
- **`Fwd Header Length.1`** is a duplicate of `Fwd Header Length`. **`CWE Flag Count`** is
  effectively constant. Drop both.
- **`Init_Win_bytes_forward` / `_backward`** are `-1` when no TCP window was observed. Preserve
  `-1` as a sentinel; do not impute it to zero or to the mean.

Do not silently build a live-inference pipeline that cannot actually work.

## 2. CICFlowMeter semantics your extractor must match

If your live extractor disagrees with CICFlowMeter on units or boundaries, the model receives
out-of-distribution input and the predictions are noise. Match these exactly, and put each one
in a comment where it is implemented:

| Property | Required behaviour |
|---|---|
| Time unit | **Microseconds**, not seconds |
| Flow timeout | 120 s of inactivity closes the flow |
| Active/idle threshold | 5 s |
| Subflow gap | 1 s |
| TCP termination | Close on RST, or on FIN seen in **both** directions |
| Packet length stats | Computed over the **L4 payload length**, not the frame length |
| Standard deviation | **Sample** std, `ddof=1` — matches Apache Commons `SummaryStatistics` |
| `Down/Up Ratio` | **Integer** division |
| Flow direction | First packet seen defines the forward direction |

Use **incremental (Welford) statistics** so memory stays constant per flow regardless of how
many packets it carries. Do not buffer packet lists.

## 3. Backend modules to build

Match these to my synopsis exactly — the module numbering is what I submit.

**Module 1 — Data Acquisition and Preprocessing** (`src/data/`, `src/preprocessing/`)
- Load the CIC-IDS2017 CSVs. Standardise column names (they have leading spaces and
  inconsistent casing in the published files). Detect the label column rather than assuming it.
- The full dataset is ~2.8 M rows. Implement **two-pass stratified sampling**: pass one counts
  labels per file, pass two samples proportionally, so a capped subset preserves class
  distribution. Memory must stay bounded.
- Normalise label text carefully. Note that the published files contain a CP-1252 `\x96` dash in
  some labels. Repair that, but **do not rewrite ASCII hyphens** — `SSH-Patator` must not become
  `SSH - Patator`.
- Remove duplicates, handle `inf` / `NaN`, encode labels.
- Split into train / validation / test **stratified**.
- Preprocessing pipeline: winsorise → impute → scale. **Fit on the train split only.** Persist
  the fitted pipeline together with its `input_features` order, and enforce that order on load.
- Produce a dataset profile: row counts, class counts, missing-value counts, per-feature summary.

**Module 2 — Feature Extraction** (`src/features/`, `src/network/feature_extractor.py`)
- The contract from section 1, plus the flow assembler that computes it from packets.
- One flow key: (src IP, src port, dst IP, dst port, protocol), bidirectional.

**Module 3 — Model Training and Selection** (`src/training/`)
- Train at least: Random Forest, Decision Tree, Logistic Regression, and XGBoost if installed.
- Handle class imbalance with **balanced class weights**, not SMOTE — synthetic minority
  oversampling on flow statistics produces records that could not exist on a real network.
- Select the best model by **measured validation macro-F1**, with a documented tie-break on
  inference latency when scores are within a configurable margin.
- Save: model, preprocessing pipeline, label encoder, feature order, and the measured metrics.
- Evaluate on the held-out test set: accuracy, precision, recall, F1, confusion matrix,
  per-class classification report. Write results to `reports/`.
- Careful: `XGBClassifier.__init__` is `(self, *, objective=..., **kwargs)`. If you filter
  hyperparameters with `inspect.signature`, you will silently drop every one of them. Detect
  `VAR_KEYWORD` and pass everything through.

**Module 4 — Live Traffic Capture** (`src/network/`)
- Scapy sniffing on a selectable interface, read-only. Enumerate interfaces. Degrade with a
  clear message when privileges are missing — do not crash.
- Flow manager with timeout expiry and a background monitor thread.
- Also accept a PCAP file as a traffic source.

**Module 5 — Intrusion Detection and Risk Assessment** (`src/inference/`)
- Apply the saved pipeline and model to each completed flow. Enforce the feature contract order.
- Apply a confidence threshold so weak predictions are not reported as attacks.
- **Risk score is not model confidence.** Implement them as separate, separately displayed
  quantities:
  ```
  risk = 100 * (0.55*severity + 0.30*confidence + 0.15*volume)
  ```
  where severity is a per-class weight from config, confidence is the model probability, and
  volume is log-scaled packets-per-second against a reference. Weights come from config and are
  validated to sum to 1.0 at start-up.
  A **benign** verdict uses a separate expression, `100 * 0.20 * (1 - confidence)`, which is
  bounded at 20 and therefore can never leave the LOW band. A confident benign classification
  must never raise an alert.
  Bands: LOW < 25 ≤ MEDIUM < 50 ≤ HIGH < 75 ≤ CRITICAL.
- Every risk score carries a human-readable formula string showing how it was computed.

**Module 6 — Explainable AI** (`src/xai/`)
- SHAP with the correct explainer per model type: `TreeExplainer` for Random Forest / XGBoost /
  Decision Tree, `LinearExplainer` for Logistic Regression, `KernelExplainer` otherwise.
- LIME tabular, using a background sample saved at training time.
- Strip transformer prefixes (`num__`, `cat__`, …) so the reported feature names are the ones a
  human recognises.
- An explanation engine that converts ranked feature contributions into plain-language security
  narrative — e.g. "this flow sent 847 packets in 1.2 seconds with almost no response traffic,
  which matches scanning behaviour." The narrative must be generated from the actual
  contribution values, not from a fixed template per class.
- SHAP on a single row must be fast enough for live use. Measure it and report the number.

**Module 7 — Alert and Database Management** (`src/database/`)
- SQLite with **WAL mode** (the monitor thread writes while the dashboard reads).
- Tables: flows, predictions, alerts, system events. Indexed on timestamp and risk level.
- Alerts raised when risk reaches the configured band.
- Metadata only. No payloads.

**Simulation mode** (`src/simulation/`)
- Synthesise realistic flow records per class and **run them through the real flow manager and
  the real feature extractor**, so the demo exercises the genuine pipeline rather than
  bypassing it.
- Add controlled jitter so classes overlap the way real traffic does. A fixture that yields
  1.000 on every metric is a broken fixture, not a good model.
- Also replay stored flow records from CSV.
- Preserve `source_ip`, `source_port`, `destination_ip`, `destination_port` through
  canonicalisation — these are display columns, and a naive rename will clobber them.

## 4. Configuration

One `config/config.yaml` holding: paths, dataset caps, split ratios, feature flags, model
hyperparameters, risk weights and severity map, band thresholds, capture settings, alert
threshold. Validate it at start-up and fail with a clear message, not a `KeyError` three
modules deep.

## 5. What "working properly" means — verify, don't assert

Build in this order, and **run each step before moving on**:

1. `python scripts/check_environment.py` — reports Python version, every required package and
   whether it imports, Npcap/libpcap presence, and capture privilege. Exits non-zero on a
   blocking problem.
2. `pytest -q` — unit tests for the feature contract, the flow manager, preprocessing,
   prediction, risk scoring, the database, and the XAI layer. Include an end-to-end integration
   test that goes synthetic packets → flows → features → prediction → explanation → database
   row. Tests must not require the real dataset or root privileges.
3. `python scripts/quickstart_demo.py` — trains on synthetic data and runs the full pipeline,
   so the system is demonstrable on a machine that has never downloaded CIC-IDS2017.
4. `python scripts/train_model.py` — real training on the dataset.
5. `python scripts/evaluate_model.py` — writes measured metrics to `reports/`.
6. `python scripts/test_live_capture.py` — captures for N seconds on a chosen interface and
   prints the flows it assembled, with no model involved. This isolates capture faults from
   model faults.

**Report the actual output of each.** If `pytest` shows failures, show me the failures. Do not
tell me the suite passes unless you ran it and it passed.

## 6. Order of work

1. The feature-contract analysis from section 1 — and **stop and show me your conclusions
   before implementing**.
2. Config, logging, errors, helpers.
3. Data loading, cleaning, splitting, preprocessing.
4. Flow manager and feature extractor, with tests proving they match the CICFlowMeter
   semantics in section 2.
5. Training, selection, evaluation.
6. Inference and risk.
7. XAI.
8. Database.
9. Simulation mode.
10. Scripts and the environment check.
11. Full test run, then the demo, then training.

## 7. Deliverables

- Every file complete and runnable on **Windows** (this is where I demo) and Linux.
- `requirements.txt` with pinned versions that actually resolve together.
- `README.md` covering install, Npcap setup, dataset download, training, demo mode, live mode,
  and a troubleshooting section. Every metric in it is either measured or marked as not measured.
- A short written note on the feature-contract decision, for my viva.

If you hit something that cannot be done as specified, tell me what and why, propose the
alternative, and wait — do not quietly substitute something weaker.