# Feature contract: viva note

## The problem

The model trains on CIC-IDS2017, whose features were produced offline by CICFlowMeter.
At detection time the features come from live packets. If the two disagree on a
definition, the model receives inputs unlike anything it saw in training, and its
predictions stop meaning anything, even though the test-split score still looks good.

## The decision

`src/features/contract.py` is the single source of truth. Training and live inference
both take their feature list from it, and the persisted pipeline enforces the column
order on load, so the two sides cannot drift apart.

**The rule: copy what the tool actually did, even where it looks wrong.** The aim is
for training and detection to compute each feature the same way, not for each feature
to be "correct" by its name.

## Classification of the 84 columns (plus `Label`)

| Outcome | Count | Columns |
|---|---:|---|
| Modelled: derivable | 43 | Packet and byte totals, packet sizes, rates, inter-arrival times, duration, initial windows, active data packets, down/up ratio |
| Modelled: approximation | 12 | Subflow columns (4), Active and Idle statistics (8) |
| Modelled if present | 1 | Protocol (absent from `MachineLearningCSV`) |
| Off by default: unverified | 12 | All TCP flag counters (9), Fwd/Bwd Header Length, `min_seg_size_forward` |
| Excluded: degenerate | 10 | Six Bulk columns, Bwd PSH Flags, Bwd URG Flags, `Fwd Header Length.1`, `CWE Flag Count` |
| Excluded: identifiers | 5 | Flow ID, Source IP, Source Port, Destination IP, Timestamp |
| Excluded: label leak | 1 | Destination Port (off by default; bucketed if enabled) |

The model therefore uses 55 numeric features, plus Protocol when the files include it.
Feature selection may then remove more, for example near-duplicate columns.

## Rules taken from the tool rather than from its documentation

| Behaviour | Evidence |
|---|---|
| The 120 s timeout runs from flow **start**, not from the last packet | `FlowGenerator.java` |
| The first packet counts **twice** in flow-wide length statistics | 471/471 sample rows satisfy `mean x (n+1) = average size x n` |
| `act_data_pkt_fwd` skips the first packet | 362/362 sample rows with payload in every forward packet |
| `Subflow *` equals the flow totals | 500/500 sample rows, including 73 with a gap above 1 s |
| The final active burst is never recorded | Call commented out in the source; 0 of 443 eligible sample rows have active time |
| Fwd PSH/URG only look at the first packet | `BasicFlow.java`; sample values are only 0 or 1 |
| Single-packet flows are dropped on timeout and flush | `FlowGenerator.java` |

## Trade-offs

- **Fidelity over elegance.** Some features deliberately reproduce tool quirks, so their
  names are not literally accurate. The alternative is a clean feature the model never
  saw in training.
- **Fewer features.** Dropping the flag and header columns gives up inputs that often
  score well offline. They are off because their published values could not be explained
  by real packets: PSH appears without ACK, and DNS flows show 20/32/40-byte "headers"
  where UDP's header is 8 bytes.
- **TCP close rule.** The system closes on RST or on FIN in both directions, as the
  project specification says. Research reports the 2017 tool closed on the first FIN and ignored
  RST, so flow boundaries remain a documented approximation of the published CSVs.
- **No proof of live accuracy.** Matching definitions removes one cause of failure. A
  good test score still does not show the model works on a different network.

## Limits of this evidence

- The sample figures come from 500 rows of a third-party, "augmented" copy of the dataset,
  and not from the original CIC files. Treat them as provisional.
- The CICFlowMeter source read is the current `master` branch, which may differ from the
  version that produced the 2017 CSVs.
- `python scripts/audit_dataset_columns.py` measures these on the real files and writes
  `reports/feature_contract_audit.json`. **[not yet measured: the dataset is not installed]**

## Likely viva questions

**Why not use Destination Port? It improves accuracy.** Attacks in the dataset ran against
fixed ports, so the model would learn the experiment's setup and not the attack.

**Why balanced class weights and not SMOTE?** Interpolating flow statistics creates records
that could not exist on a real network.

**Is risk the same as confidence?** No. Confidence is how sure the model is. Risk also
weighs how severe the class is and how much traffic the flow carries. A benign verdict uses
a separate formula capped at 20, so it can never raise an alert.

**Why select on macro-F1?** BENIGN dominates the data, so accuracy can look high while
rare attacks are missed. Macro-F1 counts every class equally.
