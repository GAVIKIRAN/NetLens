# NetLens

**Explainable AI-Based Real-Time Network Intrusion Detection System — with Live Monitoring Dashboard**

A complete, runnable Python application that trains machine-learning models on
CIC-IDS2017, captures live network traffic, reconstructs CICFlowMeter-compatible
flow features from it, classifies each flow, scores its risk, explains the
decision with SHAP and LIME, stores everything in SQLite, and presents it all in
a professional Streamlit security dashboard.

> **Defensive tool.** This system only *observes* traffic. It never generates,
> injects, scans or replays traffic onto a network. Attack detection is
> demonstrated with pre-recorded and synthetic flow records (DEMO mode), so the
> project can be presented and graded without touching anyone else's machine.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Features](#3-features)
4. [Technology stack](#4-technology-stack)
5. [Installation](#5-installation)
6. [Dataset setup](#6-dataset-setup)
7. [Training](#7-training)
8. [Model evaluation](#8-model-evaluation)
9. [Live monitoring](#9-live-monitoring)
10. [Demo mode](#10-demo-mode)
11. [Dashboard](#11-dashboard)
12. [Explainable AI](#12-explainable-ai)
13. [Database](#13-database)
14. [Testing](#14-testing)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitations](#16-limitations)
17. [Future enhancements](#17-future-enhancements)
18. [Command reference](#18-command-reference-windows)

---

## 1. Project overview

A Network Intrusion Detection System (NIDS) that watches network *flows* rather
than packets, decides whether each flow is benign or one of several attack
classes, and — crucially — can say **why**.

The pipeline has two halves that meet at a single, explicit contract:

```
OFFLINE  CIC-IDS2017 CSVs -> load -> clean -> profile -> split
                          -> preprocessing pipeline (fit on TRAIN only)
                          -> train RF / LogReg / SVM / XGBoost
                          -> evaluate -> select -> persist artefacts
                                                        |
                                    +-------------------+
                                    v
ONLINE   packets -> flow assembly -> FEATURE CONTRACT -> preprocessing
                 -> model -> confidence -> risk score
                 -> SHAP + LIME -> human-readable explanation
                 -> SQLite -> Streamlit dashboard
```

The **feature contract** (`src/features/contract.py`) is the single source of
truth for which features exist, what they are called and in what order they are
fed to the model. Training reads its column list from the contract; the live
flow meter emits exactly that dictionary; the predictor re-indexes every
incoming record to the persisted order and raises if anything is missing. The
two halves cannot silently drift apart.

### Why that contract matters (the central design decision)

CIC-IDS2017 was **not** produced from raw packets — it was produced by
**CICFlowMeter**, whose 78 columns are flow aggregates with very specific
semantics. Reproducing them live is possible, but only if you are honest about
which ones are reproducible. This project splits the columns into three groups:

| Group | Columns | Decision |
|---|---|---|
| **Reproducible** (the modelled features) | duration, fwd/bwd packet and byte counts, packet-length statistics, IAT statistics, rates, down/up ratio, segment sizes, subflow columns, `Init_Win_bytes_*`, `act_data_pkt_fwd`, Active/Idle statistics, protocol | **Kept.** `src/network/` reimplements CICFlowMeter's rules in Scapy: microsecond units, a 120 s timeout measured **from flow start**, a 5 s active/idle threshold, TCP termination on RST or bidirectional FIN, packet-length statistics over the **transport payload**, and sample (ddof = 1) standard deviations. It also copies the tool's quirks, such as counting the first packet twice in flow-wide length statistics. See [`docs/feature_contract_note.md`](docs/feature_contract_note.md). |
| **Unverified** | the nine TCP flag columns, `Fwd/Bwd Header Length`, `min_seg_size_forward` | **Off by default** (`features.use_unverified_columns: false`). Their published values could not be reproduced from packets in the sample checked so far, so the model does not rely on them. |
| **Degenerate** | the six `*Bulk*` columns, `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Header Length.1`, `CWE Flag Count` | **Dropped.** The Bulk and Bwd flag columns are reported as zero throughout the published CSVs, `Fwd Header Length.1` is an exact duplicate, and `CWE Flag Count` is near-constant with ambiguous semantics. `python scripts\audit_dataset_columns.py` measures this on your files. |
| **Identifiers / leakage** | `Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Timestamp`, `Destination Port` | **Dropped.** The first five identify the 2017 test-bed, not behaviour. `Destination Port` is the subtle one: CIC-IDS2017's attack classes are almost perfectly separable by port (SSH-Patator to 22, web attacks to 80), so a model that sees it learns *the port*, scores beautifully offline, and collapses on your network. It is **off by default** (`features.use_destination_port: false`) and, if you switch it on, it is bucketed into well-known / registered / dynamic rather than used raw. |

Every one of these decisions, with its reason, is visible in the dashboard under
**Model Performance -> Feature contract**.

---

## 2. Architecture

```
nids_xai/
├── app.py                          Streamlit entry point (page router)
├── requirements.txt
├── README.md
├── pytest.ini
├── .gitignore
├── .streamlit/config.toml          dashboard theme
│
├── config/
│   └── config.yaml                 every tunable setting, fully commented
│
├── data/
│   ├── raw/                        <- put the CIC-IDS2017 CSVs here
│   ├── processed/
│   ├── demo/                       demo_flows.csv (safe replay records)
│   └── README.md                   dataset download + layout instructions
│
├── models/
│   ├── trained/                    best_model.joblib, per-candidate models,
│   │                               model_metadata.json, evaluation_metrics.json,
│   │                               xai_background.npz
│   └── preprocessing/              preprocessing_pipeline.joblib,
│                                   label_encoder.joblib, feature_contract.json
│
├── reports/                        dataset_profile.json, model_comparison.csv,
│                                   model_selection.csv
│
├── src/
│   ├── config/settings.py          YAML + NIDS_* environment overrides
│   ├── features/contract.py        * the feature contract (single source of truth)
│   ├── data/
│   │   ├── loader.py               recursive CSV discovery, two-pass stratified load
│   │   ├── cleaner.py              NaN/Inf/duplicates/labels/rare-class merging
│   │   ├── profiler.py             dataset statistics report
│   │   └── splitter.py             stratified train / validation / test split
│   ├── preprocessing/
│   │   ├── scaler.py               quantile winsorising + scaler factory
│   │   ├── encoder.py              protocol & port encoders, label codec
│   │   ├── feature_selector.py     variance / correlation / importance filters
│   │   └── pipeline.py             * the persisted, order-enforcing pipeline
│   ├── training/
│   │   ├── train.py                end-to-end training run + artefact persistence
│   │   ├── evaluate.py             full metric suite with explicit averaging
│   │   └── model_selection.py      candidate factory + data-driven selection
│   ├── inference/
│   │   ├── predictor.py            * the only component that touches the model
│   │   └── risk.py                 documented risk formula (not confidence)
│   ├── network/
│   │   ├── interfaces.py           interface enumeration & resolution
│   │   ├── packet_capture.py       read-only Scapy sniffer + PCAP reader
│   │   ├── flow_manager.py         bidirectional flow assembly (CICFlowMeter rules)
│   │   ├── feature_extractor.py    packet parsing + flow -> contract features
│   │   └── monitor.py              LIVE / DEMO / PCAP orchestration
│   ├── simulation/
│   │   ├── synthetic.py            in-memory packet simulation (no network)
│   │   └── replay.py               demo dataset builder + replay engine
│   ├── xai/
│   │   ├── shap_explainer.py       model-appropriate SHAP explainer
│   │   ├── lime_explainer.py       LIME tabular explainer
│   │   ├── local_llm.py            optional local AI review through Ollama
│   │   └── explanation_engine.py   technical -> security narrative
│   ├── database/
│   │   ├── schema.py               flows / predictions / alerts / system_events
│   │   └── database.py             thread-safe SQLite access layer
│   └── utils/
│       ├── logger.py               rotating file + console logging
│       ├── errors.py               actionable exception hierarchy
│       └── helpers.py              formatting, normalisation, statistics
│
├── dashboard/
│   ├── state.py                    cached settings / db / monitor / model
│   ├── styles/                     theme.py (validated palette) + theme.css
│   ├── components/                 metrics, charts, tables, refresh, compat
│   └── pages_impl/                 the seven dashboard pages
│
├── scripts/
│   ├── download_dataset.py         fetch + install CIC-IDS2017 into data/raw
│   ├── quickstart_demo.py          synthetic end-to-end setup in one command
│   ├── train_model.py              train on CIC-IDS2017
│   ├── evaluate_model.py           print the stored metrics
│   ├── profile_dataset.py          dataset statistics without training
│   ├── make_demo_data.py           build the safe demo replay file
│   ├── list_interfaces.py          show capture interfaces
│   ├── test_live_capture.py        capture-path smoke test
│   ├── audit_dataset_columns.py    measure the contract's facts on the real CSVs
│   ├── measure_xai_latency.py      time SHAP and LIME per flow
│   └── check_environment.py        environment doctor
│
├── docs/feature_contract_note.md   the feature-contract decision, for the viva
├── tests/                          140 pytest tests incl. an integration test
└── logs/                           rotating nids.log
```

---

## 3. Features

**Data and training**

* Recursive CSV discovery — no filename is hard-coded, sub-folders work.
* Two-pass **stratified** loading: pass 1 counts classes per file, pass 2 samples
  each class *in proportion to where it occurs*, so the BENIGN quota is not drawn
  entirely from Monday's capture.
* Handles the real quirks of the published files: header inconsistencies,
  `Infinity` in the rate columns, NaNs, duplicate rows, repeated header rows and
  the Windows-1252 dash inside `Web Attack – Brute Force`.
* Dataset profile report: record count, feature count, class distribution,
  imbalance ratio, missing-value report, duplicate report and per-feature
  descriptive statistics — all measured, never estimated.
* Leak-free preprocessing: every learned statistic (clip bounds, imputation
  values, scaler parameters, selected features) is fitted on the **training
  split only** and then applied unchanged everywhere else.
* Five candidate models (Random Forest, Decision Tree, Logistic Regression,
  XGBoost and an optional calibrated Linear SVM) with balanced class weighting.
* Model selection from **measured validation results**, with a documented
  speed tie-break. Nothing is hard-coded as "best".

**Detection and explanation**

* Live packet capture (Scapy, read-only) with configurable interface and BPF
  filter, plus offline PCAP replay.
* Bidirectional flow assembly with CICFlowMeter-compatible semantics and
  incremental (Welford) statistics, so a million-packet flow costs the same
  memory as a three-packet one.
* Configurable attack-probability threshold, so `argmax` alone never raises an
  alert.
* Risk scoring with an explicit, printed formula — deliberately **not** the same
  thing as model confidence, and displayed separately everywhere.
* SHAP with the right explainer per model family, LIME for local surrogate
  explanations, and a narrative generator that turns weights into security
  English without ever claiming causation.

**Operations**

* SQLite persistence of flows, predictions, alerts and system events (metadata
  only — no packet payload is ever written to disk).
* Seven-page Streamlit dashboard with live auto-refresh, filtering, CSV export
  and a table view behind every chart.
* Actionable error handling: missing dataset, unreadable CSV, missing features,
  incompatible model, missing adapter, permission denied, capture failure,
  empty flow, SHAP/LIME failure, database error, corrupted model, missing config
  — each produces a specific message plus a hint, with the traceback in the log.
* 140 pytest tests including a full dataset -> prediction -> XAI -> database
  integration test.

---

## 4. Technology stack

| Layer | Library |
|---|---|
| Language | Python 3.11+ |
| Machine learning | scikit-learn, XGBoost |
| Data | pandas, NumPy, SciPy |
| Explainable AI | SHAP, LIME |
| Visualisation | Plotly (dashboard), Matplotlib (optional exports) |
| Network capture | Scapy (+ Npcap driver on Windows) |
| Dashboard | Streamlit |
| Model persistence | joblib |
| Storage | SQLite (WAL mode) |
| Configuration | YAML + `NIDS_*` environment variables |
| Logging | Python `logging` with rotating files |
| Testing | pytest |

---

## 5. Installation

### 5.1 Prerequisites

* **Python 3.11 or newer** — <https://www.python.org/downloads/windows/>
  (tick *"Add python.exe to PATH"* in the installer).
* **Npcap** — <https://npcap.com> — only needed for LIVE capture.
  Install with **"WinPcap API-compatible Mode"** ticked, then reboot.
  DEMO mode and everything else work without it.

### 5.2 Set up the project (Windows, PowerShell or CMD)

```bat
cd path\to\nids_xai

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PowerShell may block activation with *"running scripts is disabled"*. Fix it
> once with:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 5.3 Verify the installation

```bat
python scripts\check_environment.py
```

This prints your Python version, every required and optional package, the
resolved paths, whether a trained model exists, and how many capture interfaces
Scapy can see.

### 5.4 Try everything immediately (no dataset needed)

```bat
python scripts\quickstart_demo.py
streamlit run app.py
```

`quickstart_demo.py` fabricates a small CIC-IDS2017-*shaped* dataset in memory,
trains on it, builds the demo replay file, runs simulated flows through the whole
pipeline into a throwaway database and leaves the app ready. It exists so
you can confirm the installation works before downloading 220 MB.
**The metrics it prints describe synthetic data and say nothing about
CIC-IDS2017** — the script says so loudly too.

---

## 6. Dataset setup

### 6.1 Dataset download links

The project uses the **CIC-IDS2017 `MachineLearningCSV.zip`** archive.

- **Official CIC-IDS2017 dataset page:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Kaggle mirror:** https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

Download `MachineLearningCSV.zip` from the official dataset page (recommended), or use the Kaggle mirror. The archive is approximately 224 MB and is intentionally not stored in this GitHub repository because GitHub rejects individual files larger than 100 MB.

After downloading, install it with:

```bat
python scripts\download_dataset.py --zip C:\Users\you\Downloads\MachineLearningCSV.zip
python scripts\download_dataset.py --verify
```

### 6.2 The easy way

```bat
python scripts\download_dataset.py
```

Tries to fetch `MachineLearningCSV.zip` (~224 MB) from the CIC mirrors, extracts
the eight CSV files straight into `data\raw\`, deletes the archive and verifies
that the loader can read every file.

**This may not work.** When last tested (November 2026) the direct URLs redirected
to the UNB datasets page instead of serving the archive. If so, follow the download
steps on the [dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) in a
browser, then install the zip with **no networking at all**:

```bat
python scripts\download_dataset.py --links
python scripts\download_dataset.py --zip C:\Users\you\Downloads\MachineLearningCSV.zip
python scripts\download_dataset.py --verify
```

Other options:

```bat
python scripts\download_dataset.py --labelled   :: GeneratedLabelledFlows.zip instead
python scripts\download_dataset.py --both       :: both archives
python scripts\download_dataset.py --keep-zip   :: don't delete the archive
python scripts\download_dataset.py --dest D:\datasets\CICIDS2017
```

**Which archive?** `MachineLearningCSV.zip` is the one to use. It is the
standard distribution and everything in this project is built around it. Its one
quirk is that it omits the `Protocol` column — the feature contract detects that
and skips the feature rather than failing. `GeneratedLabelledFlows.zip` (~271 MB)
does include `Protocol`, so use `--labelled` if you want that one extra input;
its identifier columns are dropped by the cleaner either way.

Kaggle mirrors work too if you have an account and API token:

```bat
kaggle datasets download -d chethuhn/network-intrusion-dataset
python scripts\download_dataset.py --zip network-intrusion-dataset.zip
```

### 6.3 Or do it by hand

Download from <https://www.unb.ca/cic/datasets/ids-2017.html> and copy the CSVs
into `data\raw\`. The file list and what the loader does with them are in
[`data/README.md`](data/README.md).

Keeping the dataset elsewhere is fine:

```bat
python scripts\train_model.py --data-dir D:\datasets\CICIDS2017
```

or permanently:

```bat
set NIDS_PATHS__RAW_DATA_DIR=D:\datasets\CICIDS2017
```

Inspect the data before training anything:

```bat
python scripts\profile_dataset.py
python scripts\audit_dataset_columns.py
```

The first prints the record count, the feature count, the class distribution, the
duplicate and missing-value reports and the imbalance ratio, and writes the full
statistics to `reports\dataset_profile.json`. The second measures the facts the
feature contract relies on (constant columns, subflow and active-time behaviour,
flag values and the `-1` window sentinel) and writes
`reports\feature_contract_audit.json`. **Run it once on the real files**, because
those rules are so far based on a 500-row third-party sample.

---

## 7. Training

```bat
python scripts\train_model.py
```

Options:

```bat
python scripts\train_model.py --data-dir D:\datasets\CICIDS2017
python scripts\train_model.py --max-rows 100000
python scripts\train_model.py --quick
```

What happens, in order:

1. **Load** — two-pass stratified subsample (caps in `config.yaml -> dataset:`).
2. **Clean** — drop identifier/degenerate columns, canonicalise labels, replace
   `Infinity` with `NaN`, drop duplicates, merge rare classes.
3. **Profile** — write `reports\dataset_profile.json`.
4. **Resolve features** — intersect the contract with the columns this
   distribution of CIC-IDS2017 actually has. *(The `MachineLearningCVE`
   distribution omits `Protocol`; that is detected and reported, not fatal.)*
5. **Split** — stratified 64 / 16 / 20 train / validation / test.
6. **Fit preprocessing on TRAIN only** — winsorise, impute, scale for numeric
   columns; one-hot for categorical ones; plus variance / correlation / optional
   importance feature selection.
7. **Train** every enabled candidate with balanced class weights.
8. **Evaluate** each on the **validation** split.
9. **Select** the winner from those measurements (`model_selection.primary_metric`,
   default `macro_f1`, with a latency tie-break).
10. **Evaluate on TEST once**, after selection, for the honest estimate.
11. **Persist** every artefact listed in [Architecture](#2-architecture).

Class imbalance is handled with **balanced class weights** (and per-sample
weights for XGBoost) rather than SMOTE: synthetic points interpolated in a
60-dimensional flow-feature space do not correspond to any physically realisable
flow, and oversampling a duplicate-heavy dataset mostly amplifies duplicates.
The strategy is configurable at `preprocessing.class_balance.strategy`.

Training time on the real dataset: **370 s (about 6 minutes)** for all four enabled
candidates on a Windows 11 laptop, on the sampled data described in
[Model evaluation](#8-model-evaluation). Logistic regression took most of it (293 s).
The SVM is `enabled: false` by default because it scales poorly with row count.

---

## 8. Model evaluation

```bat
python scripts\evaluate_model.py
python scripts\evaluate_model.py --report --confusion
```

Everything is also on the dashboard's **Model Performance** page.

**Metrics produced:** accuracy, balanced accuracy, precision / recall / F1 in
macro *and* weighted form, per-class precision / recall / F1 / support,
one-vs-rest false-positive rate per class, the full confusion matrix, the
sklearn classification report, macro one-vs-rest ROC AUC, and per-record
inference latency.

**Averaging, stated explicitly** (this is a skewed multiclass problem, so a
single "accuracy" number is close to meaningless):

* **macro** — the metric is computed per class and averaged unweighted, so
  every attack class counts as much as BENIGN. **This is the selection metric.**
* **weighted** — the same per-class metrics averaged by class support; tracks
  overall correctness and is dominated by BENIGN.
* **accuracy** — the micro average. In single-label multiclass classification
  micro-precision = micro-recall = accuracy, which is why only accuracy is shown.
* **binary security view** — every attack class collapsed into `ATTACK`, giving
  the two numbers an analyst actually acts on: **detection rate** (recall over
  all attacks) and **false-positive rate** (benign flows wrongly alerted).

`evaluate_model.py` also writes `evaluation_summary.csv`,
`classification_report.txt`, `confusion_matrix.csv` and `per_class_metrics.csv` to
`reports\`, recording which data the model was trained on.

**Measured results**

Trained on CIC-IDS2017 (`MachineLearningCVE`). The loader sampled 279,616 of the
2,830,743 rows, capping large classes at 60,000. Removing duplicates left 245,502
rows: 157,120 train, 39,281 validation and 49,101 test.

Candidates on the test split:

| Model | Accuracy | Macro F1 | Weighted F1 | Detection rate | False-positive rate |
|---|---|---|---|---|---|
| **XGBoost (selected)** | 0.9954 | 0.9166 | 0.9954 | 0.9994 | 0.0030 |
| Random forest | 0.9944 | 0.9101 | 0.9944 | 0.9983 | 0.0048 |
| Decision tree | 0.9938 | 0.8825 | 0.9939 | 0.9987 | 0.0064 |
| Logistic regression | 0.9033 | 0.7024 | 0.9136 | 0.9965 | 0.3289 |

On validation, random forest had the higher macro F1 (0.9283 against 0.9263). The
two were within the 0.005 tie-break margin, so XGBoost won on lower inference latency.

XGBoost per class on the test split:

| Class | Precision | Recall | F1 | Test flows |
|---|---|---|---|---|
| BENIGN | 0.9980 | 0.9970 | 0.9975 | 11,829 |
| Bot | 0.9653 | 0.9974 | 0.9811 | 391 |
| DDoS | 0.9995 | 0.9996 | 0.9995 | 12,000 |
| DoS GoldenEye | 0.9990 | 0.9985 | 0.9988 | 2,056 |
| DoS Hulk | 0.9993 | 0.9991 | 0.9992 | 9,346 |
| DoS Slowhttptest | 0.9905 | 0.9924 | 0.9914 | 1,046 |
| DoS slowloris | 0.9926 | 0.9926 | 0.9926 | 1,075 |
| FTP-Patator | 0.9992 | 0.9983 | 0.9987 | 1,187 |
| PortScan | 0.9995 | 0.9997 | 0.9996 | 9,090 |
| SSH-Patator | 1.0000 | 0.9969 | 0.9984 | 644 |
| Rare_Attack | 0.9000 | 0.6923 | 0.7826 | 13 |
| Web Attack - Brute Force | 0.7459 | 0.7687 | 0.7571 | 294 |
| Web Attack - XSS | 0.4309 | 0.4077 | 0.4190 | 130 |

**How to read these numbers**

* **They do not predict live performance.** The test flows come from the same capture
  as the training flows and are split at random, so they share hosts and sessions.
  Traffic on another network will likely score lower, and no false-alarm rate has been
  measured on live traffic.
* **The class mix is not realistic.** BENIGN is about 24% of the test split but about
  80% of the full dataset, because it was capped. Accuracy and weighted F1 depend on
  that mix. Per-class recall and the false-positive rate depend on it much less.
* **Web attacks are the weak spot.** Most errors are the two classes being confused:
  76 of 130 XSS flows were labelled Brute Force, and 67 of 294 Brute Force flows were
  labelled XSS. They are still flagged as attacks, but the attack type is unreliable.
* **`Rare_Attack` is too small to judge.** It merges Infiltration, SQL injection and
  Heartbleed, with only 13 test flows, so its scores could move a lot on a few flows.

Timings, Windows 11 laptop:

| Measurement | Result |
|---|---|
| CIC-IDS2017 training time, four candidates | 370 s |
| XGBoost prediction, per flow in a batch | 0.007 ms |
| SHAP, one row, real XGBoost model (100 flows) | median 9.87 ms, p95 12.80 ms |
| LIME, one row, real XGBoost model (20 flows) | median 29.8 ms, p95 34.0 ms |
| Building the explainers at startup | 4.4 s |
| SHAP, one row, synthetic quickstart XGBoost (100 flows) | median 0.92 ms, p95 1.18 ms |
| LIME, one row, synthetic quickstart XGBoost (20 flows) | median 26.7 ms, p95 31.1 ms |

The real-model SHAP and LIME timings use the model trained on CIC-IDS2017, but the
flows being explained are synthetic. The synthetic quickstart model is much smaller,
so its SHAP timing is about ten times faster and should not be quoted for the real
model. No performance number in this project should be quoted unless it was measured
on the data you loaded.

---

## 9. Live monitoring

### The quick way (Windows)

Install [Npcap](https://npcap.com) with *WinPcap API-compatible Mode*, reboot, then
double-click `scripts\start_live.bat`. It checks for the driver and starts the
dashboard. In **Live Monitor** choose **LIVE**; the interface list shows only adapters
on a real network, with the default first.

### 9.1 Find your interface

```bat
python scripts\list_interfaces.py
```

Copy the exact `NAME` value into `config\config.yaml`:

```yaml
network:
  interface: "Wi-Fi"        # or "Ethernet", or a \Device\NPF_{GUID} name
```

...or select it from the dashboard's Live Monitor page, or set
`NIDS_NETWORK__INTERFACE`.

### 9.2 Test the capture path before the dashboard

```bat
python scripts\test_live_capture.py --duration 20
python scripts\test_live_capture.py --interface "Wi-Fi" --duration 30
python scripts\test_live_capture.py --pcap C:\captures\sample.pcap
python scripts\test_live_capture.py --simulate
```

Browse the web while the capture runs so there is traffic to assemble. The
script prints the flows it built and a sample of the extracted features.

### 9.3 Administrator rights

Npcap's default install lets normal users capture, so Windows usually needs no
Administrator rights. If Npcap was installed with *Restrict driver access to
Administrators only*, right-click **PowerShell** -> **Run as administrator**, then
start the dashboard from there. Otherwise Npcap raises a permission error, which the
app reports as a clear message rather than a traceback.

DEMO mode needs no privileges at all.

### 9.4 What is captured

Only flow **metadata**: addresses, ports, protocol, timings, sizes and TCP flags.
No packet payload is ever inspected, stored or logged. That is also why the
system works fine on encrypted traffic — every feature is metadata.

### 9.5 Hybrid detection

The model scores one flow at a time, which can't reveal a port scan. In a controlled
test, `nmap -sT -Pn -n -T2 -p 1-1000` left 64 stored flows on 7 ports, all BENIGN. The
causes, each confirmed against the code, the model or the training data:

* **Most probes never became flows.** A probe to a firewalled port is one SYN with no
  reply, and flows with one packet are not emitted, because CIC-IDS2017 has no such
  rows. In a simulated run, 1,000 of these probes produced 0 flows.
* **The model learned a timing that a real network doesn't reproduce.** 99% of its
  PortScan rows are one packet each way, with a median duration of 47 µs on the lab's
  wired network. The same flow scores PortScan at 47 µs but BENIGN at 1 ms.
* **A completed connection to an open port looks normal.** 89% of training rows with
  that shape are BENIGN. The model also can't see TCP flags, ports or the protocol,
  because those columns are excluded from its inputs.

Feature names, order, units, missing-value handling and label mapping were checked and
are correct, so retraining with the same data would not fix this. One capture bug was
found and fixed: Ethernet padding was being counted as payload.

**Layers**

| Layer | Component | Input | Output |
|---|---|---|---|
| 1 | XGBoost, unchanged | One completed flow | One of 13 labels, stored as is |
| 2 | `src/detection/behavior.py` | Every packet, before flow assembly | Findings per source |
| 3 | `src/detection/correlator.py` | Layers 1 and 2, plus the protocol | BENIGN, SUSPICIOUS or ATTACK, with a risk score and a reason |
| 4 | `src/network/monitor.py` | Layer 3 | Database rows and alerts |

**Behaviour rules.** All thresholds are under `detection.behavior` in `config\config.yaml`.

| Finding | Requires, within the window | Decision |
|---|---|---|
| Port scan | 25+ ports on one host, and 60%+ of attempts failed | SUSPICIOUS, ATTACK from 100 ports |
| Host sweep | 20+ hosts on one port, and 80%+ failed | SUSPICIOUS, ATTACK from 100 hosts |
| Repeated failed connections | 20+ failures to one service, and 80%+ failed | SUSPICIOUS |
| Repeated connection attempts | 60+ new connections to one service | SUSPICIOUS |
| High connection rate | 20+ new connections per second | SUSPICIOUS |
| SYN flood pattern | 100+ SYNs per second to one service, and 80%+ unanswered | ATTACK |
| Traffic burst | 3,000+ packets per second, with a mean payload under 100 bytes | SUSPICIOUS |

**Final decision.** The model's label is never overwritten.

* A model attack label with no behavioural support is ATTACK at a score of 0.90 or
  above, and SUSPICIOUS below that.
* A model attack label on a UDP or ICMP flow is kept for the record but not used. Every
  attack row in CIC-IDS2017 has a TCP window value, so the model has never seen an
  attack over those protocols. This is what stops SSDP and mDNS replies being called
  DoS Slowhttptest. UDP on port 1900 isn't whitelisted: behaviour rules still apply.
* A flow takes on a behavioural decision only if it is part of the pattern, so a shared
  source address isn't enough.

**Risk**, for a flow: `max(model risk, behaviour risk) + 10 if both agree`, where
behaviour risk is `100 * severity * (0.6 + 0.4 * strength) + repeat bonus`. Each row
stores its formula. Risk is a priority for an analyst, not the probability of an attack.

**Evaluation.** `python scripts\evaluate_hybrid.py` runs 17 synthetic packet scenarios
with 10 variations each, using seeds not used during development and thresholds fixed
beforehand. It writes `reports\hybrid_evaluation.json`.

| Decision compared with truth, 170 runs | Hybrid | Model alone |
|---|---|---|
| Accuracy | 0.882 | 0.471 |
| Precision | 1.000 | 0.000 |
| Recall | 0.778 | 0.000 |
| F1 | 0.875 | 0.000 |
| False-positive rate | 0.000 | 0.000 |
| False-negative rate | 0.222 | 1.000 |

| Truth (rows) by decision (columns) | BENIGN | SUSPICIOUS | ATTACK |
|---|---|---|---|
| BENIGN | 80 | 0 | 0 |
| SUSPICIOUS | 10 | 40 | 0 |
| ATTACK | 10 | 0 | 30 |

All 20 misses are the two scenarios built to fall below the thresholds: a scan of 10–19
ports, and a scan slow enough (one probe per 15 s) to never fill the window.

**These numbers check the logic. They are not real-world accuracy.** The same person
wrote the scenarios and the thresholds, so the scenarios fit the rules. Real performance
has to be measured with controlled traffic on your own network.

**One real capture** (5 minutes on Wi-Fi, with thresholds unchanged, 5,982 packets and
151 flows). A scan from `192.168.1.44` to `192.168.1.35` was running at the time. It was
raised as SUSPICIOUS at 26 ports and as ATTACK at 100, peaking at 137 ports with about
90% of attempts unanswered. The model labelled all 55 of that source's scored flows
BENIGN, and they stayed so labelled, with the decision recorded separately. Four early
flows stayed BENIGN overall, because they were scored before the first finding. No other
device triggered a behaviour rule. Of the other 96 flows, one was flagged: a 169-packet
flow to port 443 that the model labelled `Rare_Attack`, which is probably a false alarm.
One session is not a measured detection or false-alarm rate.

**Limitations**

* Slow scans and small scans are missed, as shown above. Widening the window finds
  more of them, at the cost of more memory and more false alarms.
* Only TCP scans are covered. UDP, SYN-less (FIN, NULL, Xmas) and IPv6 scans aren't,
  and `network.bpf_filter` excludes IPv6 by default.
* A SYN flood is judged per source, so a flood from spoofed addresses is not seen as
  one, and Wi-Fi capture may miss traffic between other devices.
* Flows scored before a finding first fires keep their earlier decision.
* Model labels on TCP flows remain unverified on live traffic and can be false alarms.
  Two TLS keep-alive flows labelled PortScan are SUSPICIOUS, not cleared.
* The older rows in the database keep their original decisions.

**Verify it**

```bat
:: On Windows, in LIVE mode from the dashboard, then after the test:
.venv\Scripts\python.exe scripts\show_detections.py --source 192.168.1.44 --minutes 15
```

```bash
# On your own Ubuntu machine, against your own Windows machine only:
nmap -sT -Pn -n -T2 -p 1-1000 192.168.1.35      # expect: port scan
for i in $(seq 1 30); do nc -z -w1 192.168.1.35 81; sleep 1; done   # expect: repeated failed connections
for i in 1 2 3; do nc -z -w1 192.168.1.35 81; done                  # expect: nothing
```

### 9.6 Attack self-test script and test matrix

`scripts\demo_attack_selftest.py` generates **real TCP attack traffic against your own
host** (no nmap/nc needed, cross-platform). It is guarded to loopback or a private-LAN
address only — it refuses any public target, so it cannot be pointed at hosts you do not
own. Capture the interface the traffic actually crosses: the **Loopback** adapter for a
`127.0.0.1` test, or **Wi-Fi/Ethernet** when the attacker is a second device.

```bat
:: port scan  -> port_scan rule (SUSPICIOUS >=25 ports, ATTACK >=100, >=60% failed)
.venv\Scripts\python.exe scripts\demo_attack_selftest.py
:: connection burst -> failed_connections / repeated_attempts / connection_rate rules
.venv\Scripts\python.exe scripts\demo_attack_selftest.py --mode burst
:: from a second device you own, captured over Wi-Fi:
python demo_attack_selftest.py --target 192.168.1.34
```

**Test matrix (what this NIDS actually does).** Detection of scans/floods comes from the
behaviour layer (TCP only); the model classifies completed TCP flows. UDP and ICMP are
not covered — this is a training-data limitation, not a bug (see §9.5 and Limitations).

| ID | Scenario | Expected here | By | Note |
|---|---|---|---|---|
| T01 | Normal web browsing | BENIGN | model | |
| T02 | DNS requests | BENIGN | — | UDP, not tracked |
| T03 | ICMP ping | BENIGN | — | ICMP, not tracked |
| T04 | TCP to local service | BENIGN | model | loopback won't appear on a Wi-Fi capture |
| T05 | **TCP** port scan | SUSPICIOUS -> ATTACK | behaviour | model may label per-flow BENIGN; final decision is ATTACK |
| T06 | **UDP** port scan | BENIGN (not supported) | — | corrected: UDP not covered |
| T07 | Repeated connection attempts | SUSPICIOUS | behaviour | must be TCP and actually fail (`--mode burst`) |
| T08 | High-rate SYN | ATTACK only if >=100 SYN/s & >=80% unanswered | behaviour | needs a real rate and an unanswering target |
| T09 | High-rate **UDP** | BENIGN (not supported) | — | corrected: UDP not covered |
| T10 | Repeated failed logins | Unreliable | model or behaviour | app-layer login failures over a completed TCP connection are not "failed connections" |
| T11 | Malformed / single-packet probes | usually BENIGN/ignored | — | one-packet probes never become a flow |
| T12 | Mixed benign + attack | both correctly, **TCP attacks only** | model + behaviour | UDP/ICMP attacks stay BENIGN |

If everything reads BENIGN in a live test, check first that the attack traffic crossed
the **captured** interface (a scan of `127.0.0.1` never crosses Wi-Fi), and that it was
**TCP**.

---

## 10. Demo mode

The presentation path, designed so that **no malicious traffic is ever
generated**. Two safe sources, in order of preference:

1. **Real CIC-IDS2017 samples.** If the dataset is present,
   `make_demo_data.py` draws a class-stratified sample of genuine flow records.
   These are measurements captured in a university lab in 2017; replaying them
   transmits nothing.
2. **Synthetic flow records.** With no dataset,
   `src/simulation/synthetic.py` fabricates packet sequences *in memory* and runs
   them through the project's own flow meter, so the resulting records are
   internally consistent (rates really are bytes divided by duration, subflow
   counts really come from the 1 s gap rule). Nine behavioural profiles are
   included: benign conversation, DDoS, DoS Hulk, DoS slowloris, PortScan,
   SSH-Patator, FTP-Patator, Bot beaconing and web brute force.

```bat
python scripts\make_demo_data.py
python scripts\make_demo_data.py --rows 6000 --synthetic --attack-ratio 0.35
```

Then in the dashboard: **Live Monitor -> mode `DEMO` -> Start**. Records stream
through the *same* predict -> risk -> explain -> persist path that live capture
uses, so a demo exercises the real system, not a mock-up.

Because the demo records carry a ground-truth label, DEMO mode stores that label
next to the model's prediction and the dashboard shows both. The demo
demonstrates the system; it does not assert that the model is always right.

A third safe mode, **PCAP**, replays a capture file you already have through the
real flow meter.

---

## 11. Dashboard

```bat
streamlit run app.py
```

Opens on <http://localhost:8501>.

| Page | Contents |
|---|---|
| **Network Security Overview** | System and monitoring status, total / benign / attack flows, attack percentage, current highest severity, average and peak risk; charts for the benign-vs-attack split, attack-type distribution, alerts and traffic over time, severity bands, traffic volume, and the top contributing features across all stored alerts; recent alert table. |
| **Live Monitor** | Mode selector (DEMO / LIVE / PCAP), interface picker, replay-rate slider, Start / Stop, a one-shot "classify 25 demo flows" button, live session counters, auto-refreshing flow table (source, destination, protocol, ports, prediction, confidence, risk, band, timestamp) and live distribution charts. |
| **Alert Center** | One filter row (severity, attack type, mode, address search, row cap), summary tiles, alerts by type and severity, the full alert table, CSV export, and a per-alert detail view with its stored explanation. |
| **AI Explanation** | Pick any stored prediction; see the class, confidence, risk band, the risk formula for that specific flow, the generated narrative, SHAP and LIME contribution charts side by side, the class-probability distribution, the feature values used, and a button to save the explanation back to the alert history. |
| **Model Performance** | Validation and test comparison tables for every candidate, macro-F1 chart, confusion matrix (raw or row-normalised), full classification report, per-class metrics, the dataset profile and cleaning report, the complete feature contract with exclusion reasons, and on-demand global SHAP importance. |
| **Traffic Analytics** | Protocol distribution, top sources by volume, packet volume over time, attack trend, flow-duration distribution, predicted-class distribution, the raw flow table and CSV export. |
| **System Settings** | Interface, BPF filter, capture duration, flow timeout, activity threshold, refresh interval, default mode, attack-probability threshold, alert severity floor, path and artefact status, database counters, pruning, clearing, and the effective configuration. |

Every chart has a table view behind it, the colour palette is validated for
colour-vision deficiency, and severity colours are always paired with their band
name so nothing depends on colour alone. On a new database, the Overview page offers
a one-click **Start demo traffic** button. The dashboard loads the Inter font from
Google Fonts and falls back to your system font when offline, so it also works
without an internet connection.

---

## 12. Explainable AI

### 12.1 SHAP

The explainer is chosen to match the model, because using the wrong one is
either wrong or unusably slow:

| Model | Explainer | Cost |
|---|---|---|
| Random Forest, XGBoost | `shap.TreeExplainer` | exact, fast — used inline on every alert |
| Logistic Regression | `shap.LinearExplainer` | exact, fast — used inline on every alert |
| Calibrated Linear SVM | `shap.KernelExplainer` | sampled, slow — computed on demand only |

Explanations are computed in the transformed feature space (the matrix the model
actually consumes) and the column names are mapped back to their canonical
contract names, so a contribution always names a feature you can recognise.

### 12.2 LIME

`LimeTabularExplainer` is fitted on the same class-stratified background sample
that training saved (`models/trained/xai_background.npz`). LIME answers a
different question from SHAP — *"which features, if perturbed, would flip this
particular decision?"* — so showing both is informative: agreement means the
explanation is robust, disagreement means the flow sits near a boundary.

### 12.3 The explanation engine

`src/xai/explanation_engine.py` turns weights into security English. Everything
is generated from the measured contributions of the current prediction; nothing
is templated per attack class.

Example of the wording it produces (the values are illustrative, not a result):

> **DDoS detected — CRITICAL risk (risk score 90/100, model confidence 99.9 %)**
>
> The model assigned this flow to **DDoS** with 99.9 % confidence. Using
> TreeExplainer, the attribution is dominated by Flow Packets/s, Flow IAT Min and
> Fwd Header Length. These features **contributed to the model's prediction**;
> they are statistical associations learned from CIC-IDS2017 and are not evidence
> that any one of them caused the behaviour.
>
> * **Flow Packets/s** = 2,339.5 packets/s (packet rate of the flow) pushed the
>   model towards 'DDoS' [weight +4.4372]
> * **Flow IAT Min** = 51 us (shortest gap between two packets of the flow)
>   pushed the model towards 'DDoS' [weight +0.4362]

**Language policy.** A feature attribution describes the *model's reasoning*, not
physical causation. A high packet rate did not *cause* a DDoS; it *contributed to
the model classifying the flow as DDoS*. The module is built around
"contributed to", "pushed the model towards" and "reduced the model's confidence
in", and a unit test asserts that "caused the attack" never appears.

### 12.4 Risk score is not model confidence

Two different quantities, computed separately, stored separately and displayed
side by side:

```
confidence = P(predicted class)                       a property of the MODEL

risk_score = 100 * ( 0.55*severity                    a property of the SITUATION
                   + 0.30*confidence
                   + 0.15*volume )

             severity  per-class weight in [0,1] from config.yaml
             volume    log1p(flow packets/s) / log1p(reference), clipped to [0,1]

benign verdicts instead use:
             risk_score = 100 * 0.20 * (1 - confidence)   -> always <= 20 -> LOW
```

Bands: `LOW < 25 <= MEDIUM < 50 <= HIGH < 75 <= CRITICAL`.

A confident benign classification can therefore never produce a high risk score,
and a merely-confident attack classification of a low-severity class does not
outrank an uncertain classification of a critical one. The exact arithmetic for
each flow is printed on the AI Explanation page.

### 12.5 Local AI review

SHAP and LIME explain why the classifier chose a label, not whether the label is
right. The classifier never sees the protocol or the ports, so it can give a TCP-only
attack label to a UDP flow, and its explanation will still describe that label.

The **Review this prediction** button on the AI Explanation page adds a check:

1. **Code sets the verdict.** *Likely false alarm* appears only when a fact rules out
   the label, currently a TCP-only attack on a non-TCP flow. Everything else is
   *Needs review*. Neither verdict confirms an attack.
2. **A local model explains the verdict** in plain language. It can't change the
   verdict. If its text sounds reassuring about an open alert, the page warns you.

The review never changes the stored prediction, risk score or alert. Flow data only
goes to an Ollama server on this computer, and other hosts are refused.

To use it, start Ollama (`ollama serve`) and set the model at `xai.local_llm` in
`config\config.yaml`. The rest of the page works without Ollama.

**Why the model doesn't choose the verdict:** when `qwen2.5-coder:7b` chose it, it
correctly cleared the two live false alarms but also called 8 of 16 real CIC-IDS2017
attack flows false alarms. All 8 were given private source addresses for the test, and
the model treated internal traffic as safe. With code setting the verdict, those 2
false alarms were cleared and all 16 attacks stayed *Needs review*. This is a small
check, not a measured accuracy, and it only shows the rule can't clear TCP attacks on
TCP flows. The AI wording can still be wrong, so go by the verdict and the checks.

---

## 13. Database

SQLite (`data\nids.db`), created automatically, WAL journalling so the dashboard
can read while capture writes.

| Table | Holds |
|---|---|
| `flows` | timestamp, source/destination IP and port, protocol, duration, packet and byte counts, termination reason, mode, and the flow's feature vector as JSON (so any prediction can be re-explained later) |
| `predictions` | timestamp, model, predicted class, attack flag, confidence, pre-threshold argmax class and probability, risk score, risk level, the risk formula, the full probability distribution, top contributing features, the explanation text, and the demo ground-truth label when replaying |
| `alerts` | timestamp, severity, attack type, source, destination, protocol, confidence, risk score, explanation, acknowledgement flag |
| `system_events` | monitoring start/stop, mode changes, capture and database errors |

**Data minimisation.** Only connection metadata is stored. No packet payload,
hostname, username or credential material is captured — none of it is needed to
detect or explain a flow-level intrusion, and a test asserts the schema has no
column that could hold it.

Retention is configurable (`database.retention_days`, default 30) and pruning /
clearing are available on the System Settings page.

---

## 14. Testing

```bat
pytest
pytest -v
pytest tests\test_features.py -v
pytest -k "risk or contract" -v
```

140 tests covering configuration loading and validation, the feature contract,
flow assembly and feature extraction, dataset loading and cleaning, the
preprocessing pipeline, the risk engine, model artefacts, the predictor, the
database, SHAP, LIME, the explanation engine and the dataset installer.

The integration test walks the whole system:

```
synthetic packets -> flow assembly -> contract features
    -> preprocessing -> model -> confidence -> risk score
    -> SHAP + LIME -> narrative -> SQLite (flow, prediction, alert)
```

plus a DEMO-mode run that starts the monitor, classifies real records through
the real path and asserts the database contents match the counters.

The suite generates its own CIC-IDS2017-*shaped* fixture (correct headers,
`Infinity` values, duplicate rows, the CP-1252 dash) and trains a real model on
it in a temporary directory. It never touches your `data\`, `models\` or `logs\`
directories, never opens a network interface, and takes about ten seconds.

---

## 15. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No CSV files found under ...` | The dataset is not in `data\raw`. See [`data/README.md`](data/README.md), or pass `--data-dir`. |
| `None of the contract's numeric features are present` | The CSVs are not the CIC-IDS2017 *flow* files (you may have downloaded the PCAPs or a different dataset). |
| `Only one class survived cleaning` | You loaded only `Monday-WorkingHours.pcap_ISCX.csv`, which is 100 % BENIGN. Add the other days. |
| `MemoryError` while loading | Lower `dataset.max_total_rows` / `dataset.max_rows_per_class`, or pass `--max-rows 100000`. |
| `Trained model not found` | Run `python scripts\train_model.py` (or `python scripts\quickstart_demo.py` to try the app without the dataset). |
| `Model expects N features but the pipeline produces M` | Mixed artefacts from two training runs. Delete `models\trained` and `models\preprocessing`, then retrain. |
| `Permission denied while opening the capture device` | Run the terminal **as Administrator** (Windows) or with `sudo` / `CAP_NET_RAW` (Linux). |
| `Npcap`/`WinPcap` errors, or no interfaces listed | Install Npcap from <https://npcap.com> with *WinPcap API-compatible Mode*, then reboot. |
| `Network interface 'X' was not found` | Run `python scripts\list_interfaces.py` and copy the exact name. |
| Capture runs but no flows appear | Flows are emitted when they *complete*. Generate traffic (browse the web), lower `network.flow_timeout`, or check that `network.bpf_filter` is not excluding your traffic. |
| Everything is classified as an attack on your own network | Expected domain shift — see [Limitations](#16-limitations). Raise `inference.attack_probability_threshold` (0.85-0.95) and `risk.alert_min_band`. |
| `SHAP`/`LIME` warnings on the AI Explanation page | Explanations are optional; detection continues. The message names the cause. Non-tree models fall back to the slower KernelExplainer. |
| Streamlit shows a stale page | Press **R** in the browser, or use the **Refresh now** button on Live Monitor. |
| PowerShell: *"running scripts is disabled"* | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Anything else | `logs\nids.log` has the full traceback; `python scripts\check_environment.py` diagnoses the environment. |

---

## 16. Limitations

Stated plainly, because a project report should contain them.

1. **Domain shift is real.** CIC-IDS2017 was captured on a 2017 university
   test-bed. A model trained on it will produce false positives on your network,
   however good its test-split scores look. The mitigations shipped here are the
   configurable probability threshold, the alert severity floor and the risk
   score — not a claim that the numbers transfer.
2. **Reimplemented is not identical.** The flow meter follows CICFlowMeter's
   rules, but it is an independent implementation. It closes TCP flows on RST or
   FIN in both directions, while research reports the 2017 tool closed on the first
   FIN and ignored RST. Distribution differences remain, and they are one more
   reason for point 1. Recording a
   baseline of your own benign traffic and tuning the threshold against it is the
   honest way to deploy this.
3. **Twenty-eight CIC-IDS2017 columns are deliberately not modelled by default**
   (identifiers, the zero-valued columns, the duplicated header-length column,
   destination port, and the unverified flag and header columns). Published papers
   that use all 78 columns are not directly comparable. Some of the rules behind
   this rest on a 500-row third-party sample until `audit_dataset_columns.py` has
   been run on the real files.
4. **Flows are classified only when they complete.** A flow is scored when it ends
   or 120 seconds after it started, whichever is first; a longer connection continues
   as a new flow. Lowering `network.flow_timeout` trades feature quality for latency.
   Single-packet flows are dropped unless closed by RST or FIN, as in CICFlowMeter,
   so a lone probe that gets no reply is not scored.
5. **Mid-session captures lose the handshake**, so `Init_Win_bytes_*` uses
   CICFlowMeter's `-1` sentinel for flows that were already open when capture
   started.
6. **No packet payload is inspected**, so payload-signature attacks that look
   statistically ordinary at the flow level will not be detected. That is a
   deliberate privacy/robustness trade-off, not an oversight.
7. **Single-host monitoring.** One interface, one machine. There is no
   distributed collection, no NetFlow/IPFIX ingestion and no clustering.
8. **Rare classes stay rare.** Heartbleed has 11 flows in the whole dataset;
   classes below `dataset.min_samples_per_class` are merged into `Rare_Attack`
   because per-class metrics on 11 samples are not meaningful.
9. **SHAP on non-tree models is slow.** KernelExplainer is sampled, so those
   explanations are computed on demand rather than inline.
10. **Encrypted traffic is handled at the metadata level only** — which is all
    this system claims to do.
11. **The dashboard is not multi-user.** It has no authentication and assumes a
    single trusted operator on localhost. Do not expose it to a network.

---

## 17. Future enhancements

* **Baseline calibration mode** — record N hours of your own benign traffic,
  measure the false-positive rate, and auto-tune the threshold against it.
* **Incremental / online learning** so the model adapts to the monitored network
  instead of staying frozen at 2017.
* **Concept-drift detection** comparing live feature distributions against the
  training distribution and warning when they diverge.
* **Newer datasets** (CSE-CIC-IDS2018, CIC-IDS2019, UNSW-NB15) behind the same
  contract, and cross-dataset evaluation to quantify transfer.
* **Analyst feedback loop** — mark an alert as a false positive in the Alert
  Center and feed those labels into the next training run.
* **Anomaly-detection second opinion** (Isolation Forest / autoencoder) to flag
  traffic unlike anything in the training set, which a closed-set classifier
  cannot do.
* **Distributed collection** — lightweight sensors shipping flow records to a
  central classifier, with NetFlow/IPFIX ingestion.
* **Alert routing** to email, Slack or syslog, and export in a SIEM-friendly
  format.
* **Authentication and role-based access** for the dashboard.
* **Packaging** as a Windows service plus an installer, so it survives reboots.

---

## 18. Command reference (Windows)

Complete sequence from a fresh checkout:

```bat
:: ---- 1. environment ---------------------------------------------------
cd path\to\nids_xai
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\check_environment.py

:: ---- 2a. try it immediately, without the dataset ----------------------
python scripts\quickstart_demo.py
streamlit run app.py

:: ---- 2b. or do it properly with CIC-IDS2017 ---------------------------
python scripts\download_dataset.py          :: ~224 MB into data\raw
python scripts\profile_dataset.py
python scripts\train_model.py
python scripts\evaluate_model.py --report --confusion
python scripts\make_demo_data.py

:: ---- 3. tests ---------------------------------------------------------
pytest

:: ---- 4. capture check (Administrator terminal) ------------------------
python scripts\list_interfaces.py
python scripts\test_live_capture.py --duration 20

:: ---- 5. launch the dashboard ------------------------------------------
streamlit run app.py
::      -> Live Monitor -> mode DEMO  -> Start      (safe, no capture)
::      -> Live Monitor -> mode LIVE  -> Start      (needs Administrator)
```

Useful variants:

```bat
python scripts\train_model.py --data-dir D:\datasets\CICIDS2017 --max-rows 200000
python scripts\train_model.py --quick
python scripts\make_demo_data.py --rows 6000 --synthetic
python scripts\download_dataset.py --links
python scripts\download_dataset.py --zip C:\Downloads\MachineLearningCSV.zip
python scripts\download_dataset.py --verify
python scripts\test_live_capture.py --interface "Wi-Fi" --duration 30
python scripts\test_live_capture.py --pcap C:\captures\sample.pcap
python scripts\test_live_capture.py --simulate
streamlit run app.py --server.port 8502
pytest -k "contract or risk" -v
set NIDS_NETWORK__INTERFACE=Ethernet
set NIDS_INFERENCE__ATTACK_PROBABILITY_THRESHOLD=0.85
```

Linux / macOS equivalents: `python3 -m venv .venv && source .venv/bin/activate`,
forward slashes in paths, and `sudo` (or `setcap cap_net_raw,cap_net_admin+eip`
on the Python binary) instead of an Administrator terminal.

---

## Ethics and intended use

This is a **defensive** monitoring tool built for an academic project. It
contains no exploit code, no malware, no credential harvesting, no packet
injection, no denial-of-service generation and no offensive scanning — and the
capture path is read-only by construction, importing none of Scapy's sending
functions.

Only monitor networks you own or are explicitly authorised to monitor. Capturing
traffic on a network without permission is unlawful in most jurisdictions, even
when only metadata is recorded.

**Dataset citation.** Iman Sharafaldin, Arash Habibi Lashkari and Ali A.
Ghorbani, *"Toward Generating a New Intrusion Detection Dataset and Intrusion
Traffic Characterization"*, 4th International Conference on Information Systems
Security and Privacy (ICISSP), Portugal, January 2018.
#