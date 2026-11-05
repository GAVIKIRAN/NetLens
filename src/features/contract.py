"""The canonical feature contract: CIC-IDS2017 <-> live network capture.

This module is the **single source of truth** for which network-flow features
the system uses.  Both sides of the system read from it:

* the *training* pipeline selects its columns from :func:`resolve_features`;
* the *live* flow meter (:mod:`src.network.feature_extractor`) emits exactly
  the names defined here.

Why a contract is necessary
---------------------------
CIC-IDS2017 was not produced from raw packets — it was produced by
**CICFlowMeter**, whose 78 columns are aggregates with very particular
semantics (microsecond units, a 120 s flow timeout, a 5 s active/idle
threshold, a 1 s subflow gap, and packet-length statistics computed over the
*transport payload* rather than the frame).  Three groups of columns then
behave very differently when you try to recompute them from a live capture:

1. **Reproducible** — timing, counting, size and TCP-flag aggregates.  These
   are recomputed by our own Scapy-based flow meter using CICFlowMeter's
   definitions.  They form the model's input space.
2. **Degenerate** — the six ``*Bulk*`` columns plus ``Bwd PSH Flags`` and
   ``Bwd URG Flags`` are reported as identically zero in the published
   CIC-IDS2017 CSVs (those eight are the "eight zero columns" in the
   literature), ``Fwd Header Length.1`` duplicates ``Fwd Header Length``, and
   ``CWE Flag Count`` is near-constant with ambiguous semantics.  They carry no
   signal, so they are dropped rather than faked.  Run
   ``scripts/audit_dataset_columns.py`` to measure this on your own files.
3. **Identifiers / leakage** — ``Flow ID``, ``Source IP``, ``Destination IP``,
   ``Source Port`` and ``Timestamp`` identify the *capture session*, not the
   behaviour.  ``Destination Port`` is the subtle one: in CIC-IDS2017 the
   attack classes are almost perfectly separable by destination port
   (SSH-Patator on 22, web attacks on 80, ...), so a model that sees it learns
   the port instead of the behaviour and collapses on a different network.
   It is therefore **excluded by default** and, when explicitly enabled, is
   bucketed into service classes rather than used as a raw number.

Everything the model can consume is listed in :data:`FEATURES` with
``live_derivable=True``.  Nothing else is ever fed to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.errors import FeatureContractError
from src.utils.helpers import normalise_column
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel used by CICFlowMeter (and by us) when a TCP window size is unknown
# because the capture started after the handshake.
UNKNOWN_WINDOW = -1

# Feature groups, used for grouping in the dashboard and the explanation engine.
GROUP_VOLUME = "Volume"
GROUP_TIMING = "Timing"
GROUP_SIZE = "Packet size"
GROUP_RATE = "Rate"
GROUP_FLAGS = "TCP flags"
GROUP_HEADER = "Header / window"
GROUP_ACTIVITY = "Activity / idle"
GROUP_META = "Connection metadata"


@dataclass(frozen=True)
class FeatureSpec:
    """Definition of one canonical flow feature."""

    name: str
    """Canonical snake_case name used everywhere in this project."""

    label: str
    """Human-readable name shown in the dashboard and explanations."""

    group: str
    """Feature group (see ``GROUP_*`` constants)."""

    unit: str
    """Physical unit, used when rendering values."""

    description: str
    """One-line description used by the explanation engine."""

    cic_aliases: tuple[str, ...] = ()
    """Raw CIC-IDS2017 header spellings that map onto this feature."""

    live_derivable: bool = True
    """``True`` when our Scapy flow meter can compute this from live packets."""

    categorical: bool = False
    """``True`` when the value is a category rather than a magnitude."""

    optional: bool = False
    """``True`` when some CIC-IDS2017 distributions omit the column."""

    exclusion_reason: str = ""
    """Populated for features that are deliberately *not* modelled."""

    higher_is_suspicious: bool | None = None
    """Rough prior used only to phrase explanations, never to make decisions."""


def _f(
    name: str,
    label: str,
    group: str,
    unit: str,
    description: str,
    *,
    aliases: Sequence[str] = (),
    live: bool = True,
    categorical: bool = False,
    optional: bool = False,
    reason: str = "",
    suspicious: bool | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        label=label,
        group=group,
        unit=unit,
        description=description,
        cic_aliases=tuple(aliases),
        live_derivable=live,
        categorical=categorical,
        optional=optional,
        exclusion_reason=reason,
        higher_is_suspicious=suspicious,
    )


# ===========================================================================
#  THE CONTRACT
# ===========================================================================
FEATURES: tuple[FeatureSpec, ...] = (
    # ---------------------------------------------------------- metadata ---
    _f("protocol", "Protocol", GROUP_META, "", 
       "IP protocol number of the flow (6=TCP, 17=UDP, 1=ICMP).",
       aliases=["Protocol"], categorical=True, optional=True),
    _f("destination_port", "Destination Port", GROUP_META, "",
       "Server-side port the flow was directed at.",
       aliases=["Destination Port"], categorical=True, optional=True,
       reason="Excluded by default: near-perfect label leakage in CIC-IDS2017."),

    # ------------------------------------------------------------ volume ---
    _f("flow_duration", "Flow Duration", GROUP_TIMING, "us",
       "Time between the first and last packet of the flow.",
       aliases=["Flow Duration"], suspicious=None),
    _f("total_fwd_packets", "Total Forward Packets", GROUP_VOLUME, "packets",
       "Number of packets sent from the initiator to the responder.",
       aliases=["Total Fwd Packets"], suspicious=True),
    _f("total_backward_packets", "Total Backward Packets", GROUP_VOLUME, "packets",
       "Number of packets sent from the responder back to the initiator.",
       aliases=["Total Backward Packets"], suspicious=None),
    _f("total_length_of_fwd_packets", "Total Forward Bytes", GROUP_VOLUME, "bytes",
       "Sum of forward-direction payload bytes.",
       aliases=["Total Length of Fwd Packets"], suspicious=True),
    _f("total_length_of_bwd_packets", "Total Backward Bytes", GROUP_VOLUME, "bytes",
       "Sum of backward-direction payload bytes.",
       aliases=["Total Length of Bwd Packets"], suspicious=None),

    # ------------------------------------------------------- packet size ---
    _f("fwd_packet_length_max", "Fwd Packet Length Max", GROUP_SIZE, "bytes",
       "Largest forward payload observed.", aliases=["Fwd Packet Length Max"]),
    _f("fwd_packet_length_min", "Fwd Packet Length Min", GROUP_SIZE, "bytes",
       "Smallest forward payload observed.", aliases=["Fwd Packet Length Min"]),
    _f("fwd_packet_length_mean", "Fwd Packet Length Mean", GROUP_SIZE, "bytes",
       "Average forward payload size.", aliases=["Fwd Packet Length Mean"]),
    _f("fwd_packet_length_std", "Fwd Packet Length Std", GROUP_SIZE, "bytes",
       "Spread of forward payload sizes.", aliases=["Fwd Packet Length Std"]),
    _f("bwd_packet_length_max", "Bwd Packet Length Max", GROUP_SIZE, "bytes",
       "Largest backward payload observed.", aliases=["Bwd Packet Length Max"]),
    _f("bwd_packet_length_min", "Bwd Packet Length Min", GROUP_SIZE, "bytes",
       "Smallest backward payload observed.", aliases=["Bwd Packet Length Min"]),
    _f("bwd_packet_length_mean", "Bwd Packet Length Mean", GROUP_SIZE, "bytes",
       "Average backward payload size.", aliases=["Bwd Packet Length Mean"]),
    _f("bwd_packet_length_std", "Bwd Packet Length Std", GROUP_SIZE, "bytes",
       "Spread of backward payload sizes.", aliases=["Bwd Packet Length Std"]),

    # -------------------------------------------------------------- rate ---
    _f("flow_bytes_s", "Flow Bytes/s", GROUP_RATE, "bytes/s",
       "Throughput of the flow in bytes per second.",
       aliases=["Flow Bytes/s"], suspicious=True),
    _f("flow_packets_s", "Flow Packets/s", GROUP_RATE, "packets/s",
       "Packet rate of the flow.", aliases=["Flow Packets/s"], suspicious=True),
    _f("fwd_packets_s", "Fwd Packets/s", GROUP_RATE, "packets/s",
       "Forward-direction packet rate.", aliases=["Fwd Packets/s"], suspicious=True),
    _f("bwd_packets_s", "Bwd Packets/s", GROUP_RATE, "packets/s",
       "Backward-direction packet rate.", aliases=["Bwd Packets/s"]),

    # ------------------------------------------------- inter-arrival time ---
    _f("flow_iat_mean", "Flow IAT Mean", GROUP_TIMING, "us",
       "Average gap between consecutive packets in either direction.",
       aliases=["Flow IAT Mean"], suspicious=False),
    _f("flow_iat_std", "Flow IAT Std", GROUP_TIMING, "us",
       "Variability of the gaps between packets.", aliases=["Flow IAT Std"]),
    _f("flow_iat_max", "Flow IAT Max", GROUP_TIMING, "us",
       "Longest gap between two packets of the flow.", aliases=["Flow IAT Max"]),
    _f("flow_iat_min", "Flow IAT Min", GROUP_TIMING, "us",
       "Shortest gap between two packets of the flow.", aliases=["Flow IAT Min"]),
    _f("fwd_iat_total", "Fwd IAT Total", GROUP_TIMING, "us",
       "Total time spanned by forward packets.", aliases=["Fwd IAT Total"]),
    _f("fwd_iat_mean", "Fwd IAT Mean", GROUP_TIMING, "us",
       "Average gap between forward packets.", aliases=["Fwd IAT Mean"]),
    _f("fwd_iat_std", "Fwd IAT Std", GROUP_TIMING, "us",
       "Variability of forward packet gaps.", aliases=["Fwd IAT Std"]),
    _f("fwd_iat_max", "Fwd IAT Max", GROUP_TIMING, "us",
       "Longest gap between forward packets.", aliases=["Fwd IAT Max"]),
    _f("fwd_iat_min", "Fwd IAT Min", GROUP_TIMING, "us",
       "Shortest gap between forward packets.", aliases=["Fwd IAT Min"]),
    _f("bwd_iat_total", "Bwd IAT Total", GROUP_TIMING, "us",
       "Total time spanned by backward packets.", aliases=["Bwd IAT Total"]),
    _f("bwd_iat_mean", "Bwd IAT Mean", GROUP_TIMING, "us",
       "Average gap between backward packets.", aliases=["Bwd IAT Mean"]),
    _f("bwd_iat_std", "Bwd IAT Std", GROUP_TIMING, "us",
       "Variability of backward packet gaps.", aliases=["Bwd IAT Std"]),
    _f("bwd_iat_max", "Bwd IAT Max", GROUP_TIMING, "us",
       "Longest gap between backward packets.", aliases=["Bwd IAT Max"]),
    _f("bwd_iat_min", "Bwd IAT Min", GROUP_TIMING, "us",
       "Shortest gap between backward packets.", aliases=["Bwd IAT Min"]),

    # --------------------------------------------------------- TCP flags ---
    _f("fwd_psh_flags", "Fwd PSH Flags", GROUP_FLAGS, "count",
       "Whether the flow's first packet carried PSH (CICFlowMeter's firstPacket rule).",
       aliases=["Fwd PSH Flags"]),
    _f("bwd_psh_flags", "Bwd PSH Flags", GROUP_FLAGS, "count",
       "PSH flags seen in the backward direction.", aliases=["Bwd PSH Flags"], live=False,
       reason="Reported as always zero in CIC-IDS2017, because the tool only updates it on "
              "the first packet, which is never backward. A real live count would be "
              "out of distribution."),
    _f("fwd_urg_flags", "Fwd URG Flags", GROUP_FLAGS, "count",
       "Whether the flow's first packet carried URG (CICFlowMeter's firstPacket rule).",
       aliases=["Fwd URG Flags"]),
    _f("bwd_urg_flags", "Bwd URG Flags", GROUP_FLAGS, "count",
       "URG flags seen in the backward direction.", aliases=["Bwd URG Flags"], live=False,
       reason="Reported as always zero in CIC-IDS2017, for the same reason as Bwd PSH Flags."),
    _f("fin_flag_count", "FIN Flag Count", GROUP_FLAGS, "count",
       "Number of packets carrying the FIN flag.", aliases=["FIN Flag Count"]),
    _f("syn_flag_count", "SYN Flag Count", GROUP_FLAGS, "count",
       "Number of packets carrying the SYN flag.",
       aliases=["SYN Flag Count"], suspicious=True),
    _f("rst_flag_count", "RST Flag Count", GROUP_FLAGS, "count",
       "Number of packets carrying the RST flag.",
       aliases=["RST Flag Count"], suspicious=True),
    _f("psh_flag_count", "PSH Flag Count", GROUP_FLAGS, "count",
       "Number of packets carrying the PSH flag.", aliases=["PSH Flag Count"]),
    _f("ack_flag_count", "ACK Flag Count", GROUP_FLAGS, "count",
       "Number of packets carrying the ACK flag.", aliases=["ACK Flag Count"]),
    _f("urg_flag_count", "URG Flag Count", GROUP_FLAGS, "count",
       "Number of packets carrying the URG flag.", aliases=["URG Flag Count"]),
    _f("ece_flag_count", "ECE Flag Count", GROUP_FLAGS, "count",
       "Number of packets carrying the ECE flag.", aliases=["ECE Flag Count"]),

    # ---------------------------------------------------- header / window ---
    _f("fwd_header_length", "Fwd Header Length", GROUP_HEADER, "bytes",
       "Total transport-header bytes in the forward direction.",
       aliases=["Fwd Header Length"]),
    _f("bwd_header_length", "Bwd Header Length", GROUP_HEADER, "bytes",
       "Total transport-header bytes in the backward direction.",
       aliases=["Bwd Header Length"]),
    _f("init_win_bytes_forward", "Init Win Bytes Fwd", GROUP_HEADER, "bytes",
       "TCP receive window advertised in the first forward packet (-1 if unseen).",
       aliases=["Init_Win_bytes_forward"]),
    _f("init_win_bytes_backward", "Init Win Bytes Bwd", GROUP_HEADER, "bytes",
       "TCP receive window advertised in the first backward packet (-1 if unseen).",
       aliases=["Init_Win_bytes_backward"]),
    _f("act_data_pkt_fwd", "Active Data Packets Fwd", GROUP_HEADER, "packets",
       "Forward packets that actually carried payload.",
       aliases=["act_data_pkt_fwd"]),
    _f("min_seg_size_forward", "Min Segment Size Fwd", GROUP_HEADER, "bytes",
       "Smallest forward transport-header size observed.",
       aliases=["min_seg_size_forward"]),

    # ------------------------------------------------- aggregate size/ratio -
    _f("min_packet_length", "Min Packet Length", GROUP_SIZE, "bytes",
       "Smallest payload in the whole flow.", aliases=["Min Packet Length"]),
    _f("max_packet_length", "Max Packet Length", GROUP_SIZE, "bytes",
       "Largest payload in the whole flow.", aliases=["Max Packet Length"]),
    _f("packet_length_mean", "Packet Length Mean", GROUP_SIZE, "bytes",
       "Average payload size across the flow.", aliases=["Packet Length Mean"]),
    _f("packet_length_std", "Packet Length Std", GROUP_SIZE, "bytes",
       "Spread of payload sizes across the flow.", aliases=["Packet Length Std"]),
    _f("packet_length_variance", "Packet Length Variance", GROUP_SIZE, "bytes^2",
       "Variance of payload sizes across the flow.",
       aliases=["Packet Length Variance"]),
    _f("down_up_ratio", "Down/Up Ratio", GROUP_VOLUME, "ratio",
       "Backward packets divided by forward packets.",
       aliases=["Down/Up Ratio"], suspicious=False),
    _f("average_packet_size", "Average Packet Size", GROUP_SIZE, "bytes",
       "Mean payload size per packet in the flow.", aliases=["Average Packet Size"]),
    _f("avg_fwd_segment_size", "Avg Fwd Segment Size", GROUP_SIZE, "bytes",
       "Mean forward segment size.", aliases=["Avg Fwd Segment Size"]),
    _f("avg_bwd_segment_size", "Avg Bwd Segment Size", GROUP_SIZE, "bytes",
       "Mean backward segment size.", aliases=["Avg Bwd Segment Size"]),

    # ---------------------------------------------------------- subflows ---
    _f("subflow_fwd_packets", "Subflow Fwd Packets", GROUP_VOLUME, "packets",
       "Forward packets per subflow (subflows split on a 1 s idle gap).",
       aliases=["Subflow Fwd Packets"]),
    _f("subflow_fwd_bytes", "Subflow Fwd Bytes", GROUP_VOLUME, "bytes",
       "Forward bytes per subflow.", aliases=["Subflow Fwd Bytes"]),
    _f("subflow_bwd_packets", "Subflow Bwd Packets", GROUP_VOLUME, "packets",
       "Backward packets per subflow.", aliases=["Subflow Bwd Packets"]),
    _f("subflow_bwd_bytes", "Subflow Bwd Bytes", GROUP_VOLUME, "bytes",
       "Backward bytes per subflow.", aliases=["Subflow Bwd Bytes"]),

    # ---------------------------------------------------- activity / idle ---
    _f("active_mean", "Active Mean", GROUP_ACTIVITY, "us",
       "Average length of the bursts during which the flow was active.",
       aliases=["Active Mean"]),
    _f("active_std", "Active Std", GROUP_ACTIVITY, "us",
       "Variability of active-burst lengths.", aliases=["Active Std"]),
    _f("active_max", "Active Max", GROUP_ACTIVITY, "us",
       "Longest active burst.", aliases=["Active Max"]),
    _f("active_min", "Active Min", GROUP_ACTIVITY, "us",
       "Shortest active burst.", aliases=["Active Min"]),
    _f("idle_mean", "Idle Mean", GROUP_ACTIVITY, "us",
       "Average idle period between bursts.", aliases=["Idle Mean"]),
    _f("idle_std", "Idle Std", GROUP_ACTIVITY, "us",
       "Variability of idle periods.", aliases=["Idle Std"]),
    _f("idle_max", "Idle Max", GROUP_ACTIVITY, "us",
       "Longest idle period.", aliases=["Idle Max"]),
    _f("idle_min", "Idle Min", GROUP_ACTIVITY, "us",
       "Shortest idle period.", aliases=["Idle Min"]),

    # =====================================================================
    #  DELIBERATELY EXCLUDED  (live_derivable=False)
    # =====================================================================
    _f("flow_id", "Flow ID", GROUP_META, "", "Composite session identifier.",
       aliases=["Flow ID"], live=False,
       reason="Identifier, not behaviour: pure leakage."),
    _f("source_ip", "Source IP", GROUP_META, "", "Initiator IP address.",
       aliases=["Source IP", "Src IP"], live=False,
       reason="Identifies the CIC-IDS2017 test-bed hosts, not attack behaviour."),
    _f("destination_ip", "Destination IP", GROUP_META, "", "Responder IP address.",
       aliases=["Destination IP", "Dst IP"], live=False,
       reason="Identifies the CIC-IDS2017 test-bed hosts, not attack behaviour."),
    _f("source_port", "Source Port", GROUP_META, "", "Initiator port.",
       aliases=["Source Port", "Src Port"], live=False,
       reason="Ephemeral and effectively random; adds noise only."),
    _f("timestamp", "Timestamp", GROUP_META, "", "Flow start wall-clock time.",
       aliases=["Timestamp"], live=False,
       reason="Encodes the capture schedule; each attack ran in its own hour."),
    _f("external_ip", "External IP", GROUP_META, "", "Test-bed annotation column.",
       aliases=["External IP"], live=False,
       reason="Test-bed annotation, absent from live traffic."),
    _f("cwe_flag_count", "CWE Flag Count", GROUP_FLAGS, "count",
       "CWR flag counter as emitted by CICFlowMeter.",
       aliases=["CWE Flag Count"], live=False,
       reason="Near-constant in CIC-IDS2017 and semantically ambiguous."),
    _f("fwd_header_length_1", "Fwd Header Length (dup)", GROUP_HEADER, "bytes",
       "Duplicate of Fwd Header Length present in the published CSVs.",
       aliases=["Fwd Header Length.1"], live=False,
       reason="Exact duplicate column; keeping it double-counts one feature."),
    _f("fwd_avg_bytes_bulk", "Fwd Avg Bytes/Bulk", GROUP_VOLUME, "bytes",
       "Bulk-transfer statistic.", aliases=["Fwd Avg Bytes/Bulk"], live=False,
       reason="Identically zero throughout CIC-IDS2017."),
    _f("fwd_avg_packets_bulk", "Fwd Avg Packets/Bulk", GROUP_VOLUME, "packets",
       "Bulk-transfer statistic.", aliases=["Fwd Avg Packets/Bulk"], live=False,
       reason="Identically zero throughout CIC-IDS2017."),
    _f("fwd_avg_bulk_rate", "Fwd Avg Bulk Rate", GROUP_RATE, "bytes/s",
       "Bulk-transfer statistic.", aliases=["Fwd Avg Bulk Rate"], live=False,
       reason="Identically zero throughout CIC-IDS2017."),
    _f("bwd_avg_bytes_bulk", "Bwd Avg Bytes/Bulk", GROUP_VOLUME, "bytes",
       "Bulk-transfer statistic.", aliases=["Bwd Avg Bytes/Bulk"], live=False,
       reason="Identically zero throughout CIC-IDS2017."),
    _f("bwd_avg_packets_bulk", "Bwd Avg Packets/Bulk", GROUP_VOLUME, "packets",
       "Bulk-transfer statistic.", aliases=["Bwd Avg Packets/Bulk"], live=False,
       reason="Identically zero throughout CIC-IDS2017."),
    _f("bwd_avg_bulk_rate", "Bwd Avg Bulk Rate", GROUP_RATE, "bytes/s",
       "Bulk-transfer statistic.", aliases=["Bwd Avg Bulk Rate"], live=False,
       reason="Identically zero throughout CIC-IDS2017."),
)

#: Fast lookup by canonical name.
BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURES}

#: Every feature the live flow meter emits, in canonical order.
LIVE_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURES if spec.live_derivable
)

#: Features deliberately dropped, with the reason (rendered in the dashboard).
EXCLUDED: dict[str, str] = {
    spec.name: spec.exclusion_reason
    for spec in FEATURES
    if not spec.live_derivable
}

#: Categorical features (one-hot encoded rather than scaled).
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURES if spec.categorical and spec.live_derivable
)

#: Numeric features, in canonical order.
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURES if spec.live_derivable and not spec.categorical
)

#: Columns the flow meter can emit, but whose published CIC-IDS2017 values could
#: not be reproduced from packets in the sample checked so far.  The flag columns
#: are 0/1 and show combinations real TCP does not produce (PSH without ACK), and
#: DNS flows carry 20/32/40-byte "headers" where UDP's header is 8 bytes.  They are
#: off unless ``features.use_unverified_columns`` is set, so that the model cannot
#: lean on values that mean something different at detection time.  Re-check them
#: with ``scripts/audit_dataset_columns.py`` once the real CSVs are available.
UNVERIFIED_FEATURES: tuple[str, ...] = (
    "fwd_psh_flags", "fwd_urg_flags",
    "fin_flag_count", "syn_flag_count", "rst_flag_count", "psh_flag_count",
    "ack_flag_count", "urg_flag_count", "ece_flag_count",
    "fwd_header_length", "bwd_header_length", "min_seg_size_forward",
)

#: Section 1 classification: how faithfully a column can be computed live.
DERIVABLE = "derivable"
APPROXIMATION = "derivable with a documented approximation"
NOT_DERIVABLE = "not derivable"

#: Why each approximate column is approximate.
APPROXIMATION_NOTES: dict[str, str] = {
    **{
        name: "Published values equal the flow totals even across gaps above 1 s, so that "
              "behaviour is reproduced rather than dividing by a subflow count."
        for name in ("subflow_fwd_packets", "subflow_fwd_bytes",
                     "subflow_bwd_packets", "subflow_bwd_bytes")
    },
    **{
        name: "Follows the tool's rule of never recording the final active burst; "
              "pending confirmation on the full dataset."
        for name in ("active_mean", "active_std", "active_max", "active_min",
                     "idle_mean", "idle_std", "idle_max", "idle_min")
    },
    **{
        name: "Published values could not be reproduced from packets; excluded from the "
              "model unless features.use_unverified_columns is enabled."
        for name in UNVERIFIED_FEATURES
    },
}

_NOT_DERIVABLE: frozenset[str] = frozenset({
    "external_ip", "cwe_flag_count", "bwd_psh_flags", "bwd_urg_flags",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_avg_bulk_rate",
    "bwd_avg_bytes_bulk", "bwd_avg_packets_bulk", "bwd_avg_bulk_rate",
})


def derivability(name: str) -> str:
    """Classify a contract column as derivable, approximate or not derivable."""
    if name in _NOT_DERIVABLE:
        return NOT_DERIVABLE
    if name in APPROXIMATION_NOTES:
        return APPROXIMATION
    return DERIVABLE


#: Reverse alias map: normalised raw header -> canonical name.
ALIAS_TO_CANONICAL: dict[str, str] = {}
for _spec in FEATURES:
    ALIAS_TO_CANONICAL[_spec.name] = _spec.name
    for _alias in _spec.cic_aliases:
        ALIAS_TO_CANONICAL[normalise_column(_alias)] = _spec.name

#: Protocol categories used by the one-hot encoder (order matters).
PROTOCOL_CATEGORIES: tuple[int, ...] = (0, 1, 6, 17)
PROTOCOL_NAMES: dict[int, str] = {0: "HOPOPT/other", 1: "ICMP", 6: "TCP", 17: "UDP"}

#: Destination-port buckets used when ``features.use_destination_port`` is on.
PORT_BUCKETS: tuple[str, ...] = ("well_known", "registered", "dynamic")


@dataclass
class ResolvedFeatures:
    """The concrete feature set for one training run / one model artefact."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        """Full ordered input column list (categoricals first, as configured)."""
        return list(self.categorical) + list(self.numeric)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "all": self.all,
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "missing": list(self.missing),
            "disabled": list(self.disabled),
        }


def canonical_name(raw_column: str) -> str:
    """Map a raw CSV header onto its canonical name (or its normalised form)."""
    normalised = normalise_column(raw_column)
    return ALIAS_TO_CANONICAL.get(normalised, normalised)


def describe(name: str) -> FeatureSpec | None:
    """Return the :class:`FeatureSpec` for ``name`` if the contract knows it."""
    return BY_NAME.get(name)


def label_for(name: str) -> str:
    """Human-readable label for a canonical (or one-hot encoded) feature name."""
    base = strip_transformer_prefix(name)
    spec = BY_NAME.get(base)
    if spec is not None:
        return spec.label
    # One-hot encoded columns arrive as "protocol_6" / "destination_port_dynamic".
    for prefix, spec_name in (("protocol_", "protocol"), ("destination_port_", "destination_port")):
        if base.startswith(prefix):
            suffix = base[len(prefix) :]
            if spec_name == "protocol":
                try:
                    return f"Protocol = {PROTOCOL_NAMES.get(int(float(suffix)), suffix)}"
                except (TypeError, ValueError):
                    return f"Protocol = {suffix}"
            return f"Destination port class = {suffix}"
    return base.replace("_", " ").title()


def strip_transformer_prefix(name: str) -> str:
    """Remove the ``num__`` / ``cat__`` prefixes added by ``ColumnTransformer``."""
    text = str(name)
    for prefix in ("num__", "cat__", "proto__", "dport__", "remainder__", "passthrough__"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def resolve_features(
    available_columns: Iterable[str],
    use_protocol: bool = True,
    use_destination_port: bool = False,
    use_unverified_columns: bool = False,
) -> ResolvedFeatures:
    """Work out which contract features are usable for this dataset.

    ``available_columns`` must already be canonicalised (see
    :func:`canonical_name`).  Features that the contract knows about but the
    dataset does not contain are reported in ``missing`` — this happens for
    real: the ``MachineLearningCVE`` distribution of CIC-IDS2017 omits the
    ``Protocol`` column entirely, while ``GeneratedLabelledFlows`` includes it.
    """
    available = {str(c) for c in available_columns}
    resolved = ResolvedFeatures()

    for name in CATEGORICAL_FEATURES:
        if name == "protocol" and not use_protocol:
            resolved.disabled.append(name)
            continue
        if name == "destination_port" and not use_destination_port:
            resolved.disabled.append(name)
            continue
        if name in available:
            resolved.categorical.append(name)
        else:
            resolved.missing.append(name)

    for name in NUMERIC_FEATURES:
        if name in UNVERIFIED_FEATURES and not use_unverified_columns:
            resolved.disabled.append(name)
        elif name in available:
            resolved.numeric.append(name)
        else:
            resolved.missing.append(name)

    if not resolved.numeric:
        raise FeatureContractError(
            "None of the contract's numeric features are present in the dataset.",
            hint=(
                "The CSV headers do not look like CIC-IDS2017. Check "
                "paths.raw_data_dir and that the files are the flow-level CSVs."
            ),
        )
    if resolved.missing:
        logger.warning(
            "%d contract feature(s) absent from this dataset and skipped: %s",
            len(resolved.missing),
            ", ".join(resolved.missing),
        )
    return resolved


def port_bucket(port: float | int | None) -> str:
    """Bucket a TCP/UDP port into a coarse service class.

    Raw port numbers leak labels in CIC-IDS2017; buckets keep the (weak) signal
    "is this a server service or an ephemeral port" without memorising 22/80.
    """
    try:
        value = int(port)
    except (TypeError, ValueError):
        return PORT_BUCKETS[-1]
    if value < 0:
        return PORT_BUCKETS[-1]
    if value <= 1023:
        return PORT_BUCKETS[0]
    if value <= 49151:
        return PORT_BUCKETS[1]
    return PORT_BUCKETS[2]


def empty_record() -> dict[str, float]:
    """A zero-filled record containing every live-derivable feature."""
    record: dict[str, float] = {}
    for name in LIVE_FEATURES:
        if name in ("init_win_bytes_forward", "init_win_bytes_backward"):
            record[name] = float(UNKNOWN_WINDOW)
        else:
            record[name] = 0.0
    return record


def align_frame(
    frame: pd.DataFrame,
    expected: Sequence[str],
    *,
    strict: bool = True,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Return ``frame`` re-indexed to exactly ``expected`` columns, in order.

    This is the single guarantee that live inference never sees a different
    feature order from training.

    Parameters
    ----------
    strict:
        When ``True`` (the default for live inference) a missing column raises
        :class:`~src.utils.errors.FeatureContractError` instead of being
        silently filled.
    """
    expected = list(expected)
    missing = [c for c in expected if c not in frame.columns]
    if missing and strict:
        raise FeatureContractError(
            f"{len(missing)} expected feature(s) missing from the record: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else ""),
            hint="The model artefact and the flow meter are out of sync; retrain.",
        )
    aligned = frame.reindex(columns=expected)
    if missing:
        for column in missing:
            aligned[column] = fill_value
        aligned = aligned[expected]
    return aligned


def records_to_frame(
    records: Sequence[Mapping[str, float]], expected: Sequence[str]
) -> pd.DataFrame:
    """Build a model-ready ``DataFrame`` from flow-feature dictionaries."""
    if not records:
        return pd.DataFrame(columns=list(expected), dtype=np.float64)
    frame = pd.DataFrame(list(records))
    return align_frame(frame, expected, strict=True)


def contract_summary() -> pd.DataFrame:
    """Tabular view of the contract, rendered on the dashboard."""
    rows = []
    for spec in FEATURES:
        rows.append(
            {
                "feature": spec.name,
                "label": spec.label,
                "group": spec.group,
                "unit": spec.unit,
                "live_derivable": spec.live_derivable,
                "categorical": spec.categorical,
                "cic_ids2017_column": spec.cic_aliases[0] if spec.cic_aliases else "",
                "status": "modelled" if spec.live_derivable else "excluded",
                "derivability": derivability(spec.name),
                "reason": spec.exclusion_reason or APPROXIMATION_NOTES.get(spec.name, ""),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "FeatureSpec",
    "FEATURES",
    "BY_NAME",
    "LIVE_FEATURES",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "EXCLUDED",
    "ALIAS_TO_CANONICAL",
    "PROTOCOL_CATEGORIES",
    "PROTOCOL_NAMES",
    "PORT_BUCKETS",
    "UNKNOWN_WINDOW",
    "UNVERIFIED_FEATURES",
    "APPROXIMATION_NOTES",
    "DERIVABLE",
    "APPROXIMATION",
    "NOT_DERIVABLE",
    "derivability",
    "ResolvedFeatures",
    "canonical_name",
    "describe",
    "label_for",
    "strip_transformer_prefix",
    "resolve_features",
    "port_bucket",
    "empty_record",
    "align_frame",
    "records_to_frame",
    "contract_summary",
]
