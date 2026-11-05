"""SQLite schema definition (Phase 10).

Data-minimisation note
----------------------
Only connection *metadata* is stored - addresses, ports, protocol, timing and
volume statistics plus the model's output.  **No packet payload is ever
written to disk**, and no user, hostname or credential material is captured,
because none of it is needed to detect or explain a flow-level intrusion.
"""

from __future__ import annotations

SCHEMA_VERSION = 3

#: Added by version 2.  Existing rows keep NULL here, and old code keeps working.
PREDICTION_COLUMNS_V2: tuple[tuple[str, str], ...] = (
    ("final_classification", "TEXT"),
    ("behaviour_result", "TEXT"),
    ("detection_reason", "TEXT"),
    ("ml_risk_score", "REAL"),
    ("ml_family", "TEXT"),
)

#: Added by version 3.  Each monitoring run is one row in ``sessions`` and every
#: flow / prediction / alert / behaviour event is tagged with its ``session_id``.
#: Rows written before v3 keep NULL (shown as the "Legacy" session in the UI).
SESSION_TAGGED_TABLES: tuple[str, ...] = (
    "flows",
    "predictions",
    "alerts",
    "behaviour_events",
)

CREATE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_info (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at        TEXT    NOT NULL,
        ended_at          TEXT,
        mode              TEXT    NOT NULL DEFAULT 'LIVE',
        label             TEXT,
        flows_processed   INTEGER NOT NULL DEFAULT 0,
        attacks_detected  INTEGER NOT NULL DEFAULT 0,
        alerts_raised     INTEGER NOT NULL DEFAULT 0,
        behaviour_findings INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flows (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        flow_uid          TEXT    NOT NULL,
        timestamp         TEXT    NOT NULL,
        started_at        REAL,
        ended_at          REAL,
        src_ip            TEXT,
        dst_ip            TEXT,
        src_port          INTEGER,
        dst_port          INTEGER,
        protocol          INTEGER,
        protocol_name     TEXT,
        duration_seconds  REAL,
        packet_count      INTEGER,
        byte_count        INTEGER,
        termination       TEXT,
        mode              TEXT    NOT NULL DEFAULT 'LIVE',
        features_json     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        flow_id             INTEGER,
        timestamp           TEXT    NOT NULL,
        model               TEXT,
        prediction          TEXT    NOT NULL,
        is_attack           INTEGER NOT NULL DEFAULT 0,
        confidence          REAL    NOT NULL DEFAULT 0,
        argmax_label        TEXT,
        argmax_confidence   REAL,
        threshold_applied   INTEGER NOT NULL DEFAULT 0,
        risk_score          REAL    NOT NULL DEFAULT 0,
        risk_level          TEXT    NOT NULL DEFAULT 'LOW',
        risk_formula        TEXT,
        probabilities_json  TEXT,
        top_features_json   TEXT,
        explanation         TEXT,
        explainer           TEXT,
        mode                TEXT    NOT NULL DEFAULT 'LIVE',
        ground_truth        TEXT,
        FOREIGN KEY (flow_id) REFERENCES flows (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id     INTEGER,
        flow_id           INTEGER,
        timestamp         TEXT    NOT NULL,
        severity          TEXT    NOT NULL,
        attack_type       TEXT    NOT NULL,
        src_ip            TEXT,
        dst_ip            TEXT,
        src_port          INTEGER,
        dst_port          INTEGER,
        protocol_name     TEXT,
        confidence        REAL,
        risk_score        REAL,
        explanation       TEXT,
        top_features_json TEXT,
        acknowledged      INTEGER NOT NULL DEFAULT 0,
        mode              TEXT    NOT NULL DEFAULT 'LIVE',
        FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE CASCADE,
        FOREIGN KEY (flow_id)       REFERENCES flows (id)       ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS system_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp    TEXT NOT NULL,
        level        TEXT NOT NULL,
        component    TEXT NOT NULL,
        message      TEXT NOT NULL,
        details_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS behaviour_events (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT NOT NULL,
        kind             TEXT NOT NULL,
        decision         TEXT NOT NULL,
        src_ip           TEXT,
        dst_ip           TEXT,
        dst_port         INTEGER,
        risk_score       REAL,
        risk_level       TEXT,
        risk_formula     TEXT,
        reason           TEXT,
        indicators_json  TEXT,
        repeats          INTEGER NOT NULL DEFAULT 1,
        mode             TEXT NOT NULL DEFAULT 'LIVE'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_behaviour_timestamp ON behaviour_events (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_flows_timestamp    ON flows (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_flows_uid          ON flows (flow_uid)",
    "CREATE INDEX IF NOT EXISTS idx_pred_timestamp     ON predictions (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pred_prediction    ON predictions (prediction)",
    "CREATE INDEX IF NOT EXISTS idx_pred_is_attack     ON predictions (is_attack)",
    "CREATE INDEX IF NOT EXISTS idx_pred_flow          ON predictions (flow_id)",
    "CREATE INDEX IF NOT EXISTS idx_pred_risk_level    ON predictions (risk_level)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp   ON alerts (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_severity    ON alerts (severity)",
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp   ON system_events (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_started   ON sessions (started_at)",
)

TABLES: tuple[str, ...] = (
    "flows", "predictions", "alerts", "system_events", "behaviour_events", "sessions",
)

__all__ = [
    "CREATE_STATEMENTS",
    "PREDICTION_COLUMNS_V2",
    "SCHEMA_VERSION",
    "SESSION_TAGGED_TABLES",
    "TABLES",
]
