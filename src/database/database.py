"""SQLite persistence layer (Phase 10).

Thread-safety
-------------
The live monitor writes from a background thread while Streamlit reads from the
main thread, so each thread gets its own connection (``threading.local``) and
writes are serialised with a process-wide lock.  WAL journalling lets a reader
and the writer proceed concurrently, which keeps the dashboard responsive while
capture is running.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from src.database.schema import (
    CREATE_STATEMENTS,
    PREDICTION_COLUMNS_V2,
    SCHEMA_VERSION,
    SESSION_TAGGED_TABLES,
    SUMMARY_INDEXES,
    TABLES,
)
from src.utils.errors import DatabaseError
from src.utils.helpers import utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Sentinel distinguishing "no session filter" from "filter to session_id = NULL".
_UNSET: Any = object()


class Database:
    """All reads and writes to the NIDS store go through this class."""

    def __init__(
        self,
        path: str | Path,
        *,
        journal_mode: str = "WAL",
        timeout: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self.journal_mode = str(journal_mode)
        self.timeout = float(timeout)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        #: Rows written while a session is open are tagged with this id; None means
        #: "no session" (e.g. a one-off demo burst before start()), stored as NULL.
        self._current_session_id: int | None = None
        self.initialise()

    # ------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                str(self.path), timeout=self.timeout, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open the database at {self.path}: {exc}") from exc
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(f"PRAGMA journal_mode={self.journal_mode}")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error as exc:  # pragma: no cover - pragma support varies
            logger.warning("PRAGMA setup failed: %s", exc)
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = self._connect()
            self._local.connection = conn
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Cursor]:
        with self._write_lock:
            conn = self.connection
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                raise DatabaseError(f"Database write failed: {exc}") from exc
            finally:
                cursor.close()

    def initialise(self) -> None:
        """Create the schema if it does not exist. Safe to call repeatedly."""
        with self._write() as cursor:
            for statement in CREATE_STATEMENTS:
                cursor.execute(statement)
            existing = {row[1] for row in cursor.execute("PRAGMA table_info(predictions)")}
            for name, sql_type in PREDICTION_COLUMNS_V2:
                if name not in existing:
                    cursor.execute(f"ALTER TABLE predictions ADD COLUMN {name} {sql_type}")
            # v3: tag each data table with the session it belongs to. Old rows keep
            # NULL and surface as the "Legacy" session in the dashboard.
            for table in SESSION_TAGGED_TABLES:
                columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
                if "session_id" not in columns:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN session_id INTEGER")
            for statement in SUMMARY_INDEXES:
                cursor.execute(statement)
            cursor.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
        logger.info("Database ready at %s", self.path)

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover - defensive
                pass
            self._local.connection = None

    # -------------------------------------------------------------- writes
    def insert_flow(self, identity: Mapping[str, Any], features: Mapping[str, float],
                    *, mode: str = "LIVE", flow_uid: str = "") -> int:
        """Persist one completed flow; returns its row id."""
        with self._write() as cursor:
            cursor.execute(
                """
                INSERT INTO flows (
                    flow_uid, timestamp, started_at, ended_at, src_ip, dst_ip,
                    src_port, dst_port, protocol, protocol_name, duration_seconds,
                    packet_count, byte_count, termination, mode, features_json,
                    session_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    flow_uid or _make_uid(identity),
                    utc_now_iso(),
                    _num(identity.get("start_time")),
                    _num(identity.get("end_time")),
                    str(identity.get("src_ip", "")),
                    str(identity.get("dst_ip", "")),
                    _int(identity.get("src_port")),
                    _int(identity.get("dst_port")),
                    _int(identity.get("protocol")),
                    str(identity.get("protocol_name", "")),
                    _num(identity.get("duration_seconds")),
                    _int(identity.get("packets")),
                    _int(identity.get("bytes")),
                    str(identity.get("termination", "")),
                    str(mode),
                    # NaN means "not measurable" and is imputed by the pipeline, so it
                    # is stored as null.  As 0 it would re-score as a different flow.
                    json.dumps({k: _json_number(v) for k, v in dict(features).items()}),
                    self._current_session_id,
                ),
            )
            return int(cursor.lastrowid)

    def insert_prediction(
        self,
        flow_id: int | None,
        payload: Mapping[str, Any],
        *,
        mode: str = "LIVE",
        ground_truth: str | None = None,
    ) -> int:
        """Persist one prediction; returns its row id."""
        with self._write() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    flow_id, timestamp, model, prediction, is_attack, confidence,
                    argmax_label, argmax_confidence, threshold_applied,
                    risk_score, risk_level, risk_formula, probabilities_json,
                    top_features_json, explanation, explainer, mode, ground_truth,
                    final_classification, behaviour_result, detection_reason,
                    ml_risk_score, ml_family, session_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    flow_id,
                    utc_now_iso(),
                    str(payload.get("model", "")),
                    str(payload.get("prediction", "")),
                    1 if payload.get("is_attack") else 0,
                    _num(payload.get("confidence")),
                    str(payload.get("argmax_label", "")),
                    _num(payload.get("argmax_confidence")),
                    1 if payload.get("threshold_applied") else 0,
                    _num(payload.get("risk_score")),
                    str(payload.get("risk_level", "LOW")),
                    str(payload.get("formula", "")),
                    json.dumps(payload.get("probabilities", {})),
                    json.dumps(payload.get("top_features", [])),
                    str(payload.get("explanation", "")),
                    str(payload.get("explainer", "")),
                    str(mode),
                    ground_truth,
                    payload.get("final_classification"),
                    payload.get("behaviour_result"),
                    payload.get("detection_reason"),
                    payload.get("ml_risk_score"),
                    payload.get("ml_family"),
                    self._current_session_id,
                ),
            )
            return int(cursor.lastrowid)

    def insert_alert(
        self,
        prediction_id: int | None,
        flow_id: int | None,
        identity: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        mode: str = "LIVE",
    ) -> int:
        """Persist one alert; returns its row id."""
        with self._write() as cursor:
            cursor.execute(
                """
                INSERT INTO alerts (
                    prediction_id, flow_id, timestamp, severity, attack_type,
                    src_ip, dst_ip, src_port, dst_port, protocol_name,
                    confidence, risk_score, explanation, top_features_json, mode,
                    session_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    prediction_id,
                    flow_id,
                    utc_now_iso(),
                    str(payload.get("risk_level", "LOW")),
                    str(payload.get("prediction", "UNKNOWN")),
                    str(identity.get("src_ip", "")),
                    str(identity.get("dst_ip", "")),
                    _int(identity.get("src_port")),
                    _int(identity.get("dst_port")),
                    str(identity.get("protocol_name", "")),
                    _num(payload.get("confidence")),
                    _num(payload.get("risk_score")),
                    str(payload.get("explanation", "")),
                    json.dumps(payload.get("top_features", [])),
                    str(mode),
                    self._current_session_id,
                ),
            )
            return int(cursor.lastrowid)

    def insert_behaviour_event(
        self,
        event: Mapping[str, Any],
        *,
        mode: str = "LIVE",
    ) -> int:
        """Persist one behavioural finding; returns its row id."""
        with self._write() as cursor:
            cursor.execute(
                """
                INSERT INTO behaviour_events (
                    timestamp, kind, decision, src_ip, dst_ip, dst_port, risk_score,
                    risk_level, risk_formula, reason, indicators_json, repeats, mode,
                    session_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    utc_now_iso(),
                    str(event.get("kind", "")),
                    str(event.get("decision", "")),
                    str(event.get("src_ip", "")),
                    str(event.get("dst_ip", "")),
                    _int(event.get("dst_port")),
                    _num(event.get("risk_score")),
                    str(event.get("risk_level", "LOW")),
                    str(event.get("risk_formula", "")),
                    str(event.get("reason", "")),
                    json.dumps(dict(event.get("indicators") or {}), default=str),
                    _int(event.get("repeats")) or 1,
                    str(mode),
                    self._current_session_id,
                ),
            )
            return int(cursor.lastrowid)

    # ----------------------------------------------------------- sessions
    def start_session(self, mode: str = "LIVE", *, label: str | None = None) -> int:
        """Open a new session and make it the tag for subsequent writes."""
        with self._write() as cursor:
            cursor.execute(
                "INSERT INTO sessions (started_at, mode, label) VALUES (?,?,?)",
                (utc_now_iso(), str(mode), label),
            )
            session_id = int(cursor.lastrowid)
        self._current_session_id = session_id
        logger.info("Session %d started (mode=%s)", session_id, mode)
        return session_id

    def end_session(self, stats: Mapping[str, Any] | None = None) -> None:
        """Close the current session, storing its final counters."""
        session_id = self._current_session_id
        if session_id is None:
            return
        stats = stats or {}
        with self._write() as cursor:
            cursor.execute(
                """
                UPDATE sessions SET ended_at = ?, flows_processed = ?,
                    attacks_detected = ?, alerts_raised = ?, behaviour_findings = ?
                WHERE id = ?
                """,
                (
                    utc_now_iso(),
                    int(stats.get("flows_processed", 0) or 0),
                    int(stats.get("final_attacks", stats.get("attacks_detected", 0)) or 0),
                    int(stats.get("alerts_raised", 0) or 0),
                    int(stats.get("behaviour_findings", 0) or 0),
                    session_id,
                ),
            )
        logger.info("Session %d ended", session_id)
        self._current_session_id = None

    @property
    def current_session_id(self) -> int | None:
        return self._current_session_id

    def list_sessions(self, limit: int = 200) -> pd.DataFrame:
        """All recorded sessions, most recent first."""
        return self.query(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", [int(limit)]
        )

    def export_session(self, session_id: int | None) -> pd.DataFrame:
        """One flat, CSV-ready row per classified flow in a session (NULL = Legacy)."""
        where = "p.session_id IS NULL" if session_id is None else "p.session_id = ?"
        params: list[Any] = [] if session_id is None else [int(session_id)]
        sql = f"""
            SELECT p.timestamp, f.src_ip, f.src_port, f.dst_ip, f.dst_port,
                   f.protocol_name, f.packet_count, f.byte_count, f.duration_seconds,
                   p.prediction AS ml_prediction, p.confidence, p.is_attack,
                   p.final_classification, p.behaviour_result, p.risk_score,
                   p.risk_level, p.detection_reason, p.mode, p.ground_truth
            FROM predictions p LEFT JOIN flows f ON p.flow_id = f.id
            WHERE {where}
            ORDER BY p.id ASC
        """
        return self.query(sql, params)

    def recent_behaviour_events(
        self,
        limit: int = 100,
        *,
        mode: str | None = None,
        session_id: int | None = _UNSET,
    ) -> pd.DataFrame:
        """Recent behavioural findings. Pass ``session_id`` to scope to one session
        (``None`` selects only untagged legacy rows); omit it for all sessions."""
        clauses: list[str] = []
        params: list[Any] = []
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if session_id is not _UNSET:
            if session_id is None:
                clauses.append("session_id IS NULL")
            else:
                clauses.append("session_id = ?")
                params.append(int(session_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        return self.query(
            f"SELECT * FROM behaviour_events {where} ORDER BY id DESC LIMIT ?", params
        )

    def log_event(
        self,
        level: str,
        component: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        """Record a system event (start/stop, errors, mode changes)."""
        try:
            with self._write() as cursor:
                cursor.execute(
                    """
                    INSERT INTO system_events (timestamp, level, component, message, details_json)
                    VALUES (?,?,?,?,?)
                    """,
                    (
                        utc_now_iso(),
                        str(level).upper(),
                        str(component),
                        str(message)[:2000],
                        json.dumps(dict(details or {}), default=str),
                    ),
                )
                return int(cursor.lastrowid)
        except DatabaseError:  # events must never break the pipeline
            logger.exception("Could not persist a system event")
            return -1

    def acknowledge_alert(self, alert_id: int, acknowledged: bool = True) -> None:
        with self._write() as cursor:
            cursor.execute(
                "UPDATE alerts SET acknowledged=? WHERE id=?",
                (1 if acknowledged else 0, int(alert_id)),
            )

    def update_prediction_explanation(
        self,
        prediction_id: int,
        explanation: str,
        top_features: Sequence[Mapping[str, Any]],
        explainer: str,
    ) -> None:
        """Backfill an explanation computed on demand from the dashboard."""
        with self._write() as cursor:
            cursor.execute(
                """
                UPDATE predictions
                   SET explanation=?, top_features_json=?, explainer=?
                 WHERE id=?
                """,
                (
                    str(explanation),
                    json.dumps(list(top_features), default=str),
                    str(explainer),
                    int(prediction_id),
                ),
            )

    # --------------------------------------------------------------- reads
    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        """Run a read-only query and return a ``DataFrame``."""
        try:
            return pd.read_sql_query(sql, self.connection, params=tuple(params))
        except Exception as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    def recent_predictions(
        self,
        limit: int = 200,
        *,
        only_attacks: bool = False,
        mode: str | None = None,
    ) -> pd.DataFrame:
        clauses, params = [], []
        if only_attacks:
            clauses.append("p.is_attack = 1")
        if mode:
            clauses.append("p.mode = ?")
            params.append(mode)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        return self.query(
            f"""
            SELECT p.id, p.timestamp, p.model, p.prediction, p.is_attack,
                   p.confidence, p.risk_score, p.risk_level, p.threshold_applied,
                   p.argmax_label, p.argmax_confidence, p.explanation, p.mode,
                   p.ground_truth, p.flow_id, p.final_classification,
                   p.behaviour_result, p.detection_reason, p.ml_risk_score, p.ml_family,
                   f.src_ip, f.dst_ip, f.src_port, f.dst_port, f.protocol_name,
                   f.packet_count, f.byte_count, f.duration_seconds
              FROM predictions p
              LEFT JOIN flows f ON f.id = p.flow_id
              {where}
             ORDER BY p.id DESC
             LIMIT ?
            """,
            params,
        )

    def recent_alerts(
        self,
        limit: int = 200,
        *,
        severities: Sequence[str] | None = None,
        attack_types: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> pd.DataFrame:
        clauses, params = [], []
        if severities:
            clauses.append(f"severity IN ({','.join('?' * len(severities))})")
            params.extend(list(severities))
        if attack_types:
            clauses.append(f"attack_type IN ({','.join('?' * len(attack_types))})")
            params.extend(list(attack_types))
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        return self.query(
            f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ?", params
        )

    def get_prediction(self, prediction_id: int) -> dict[str, Any] | None:
        frame = self.query(
            """
            SELECT p.*, f.src_ip, f.dst_ip, f.src_port, f.dst_port,
                   f.protocol_name, f.features_json, f.packet_count, f.byte_count,
                   f.duration_seconds
              FROM predictions p
              LEFT JOIN flows f ON f.id = p.flow_id
             WHERE p.id = ?
            """,
            (int(prediction_id),),
        )
        if frame.empty:
            return None
        row = frame.iloc[0].to_dict()
        row["features"] = _load_json(row.get("features_json"), {})
        row["probabilities"] = _load_json(row.get("probabilities_json"), {})
        row["top_features"] = _load_json(row.get("top_features_json"), [])
        return row

    def statistics(self, mode: str | None = None) -> dict[str, Any]:
        """Headline counters for the overview page."""
        where = "WHERE mode = ?" if mode else ""
        params: tuple = (mode,) if mode else ()
        frame = self.query(
            f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN COALESCE(final_classification,
                                          CASE WHEN is_attack = 1 THEN 'ATTACK' ELSE 'BENIGN' END)
                             <> 'BENIGN' THEN 1 ELSE 0 END) AS attacks,
                   AVG(confidence) AS avg_confidence,
                   AVG(risk_score) AS avg_risk,
                   MAX(risk_score) AS max_risk
              FROM predictions {where}
            """,
            params,
        )
        row = frame.iloc[0].to_dict() if not frame.empty else {}
        total = int(row.get("total") or 0)
        attacks = int(row.get("attacks") or 0)
        alerts = self.query(
            f"SELECT COUNT(*) AS n FROM alerts {where}", params
        ).iloc[0]["n"]
        flows = self.query(
            f"SELECT COUNT(*) AS n, SUM(packet_count) AS pkts, SUM(byte_count) AS bytes "
            f"FROM flows {where}",
            params,
        ).iloc[0]
        return {
            "total_flows": total,
            "benign_flows": total - attacks,
            "attack_flows": attacks,
            "attack_percentage": round(100.0 * attacks / total, 2) if total else 0.0,
            "avg_confidence": round(float(row.get("avg_confidence") or 0.0), 4),
            "avg_risk_score": round(float(row.get("avg_risk") or 0.0), 2),
            "max_risk_score": round(float(row.get("max_risk") or 0.0), 2),
            "total_alerts": int(alerts or 0),
            "captured_flows": int(flows["n"] or 0),
            "total_packets": int(flows["pkts"] or 0),
            "total_bytes": int(flows["bytes"] or 0),
        }

    def attack_distribution(self, mode: str | None = None) -> pd.DataFrame:
        where = "WHERE is_attack = 1" + (" AND mode = ?" if mode else "")
        params: tuple = (mode,) if mode else ()
        return self.query(
            f"""
            SELECT prediction AS attack_type, COUNT(*) AS count,
                   AVG(confidence) AS avg_confidence, AVG(risk_score) AS avg_risk
              FROM predictions {where}
             GROUP BY prediction
             ORDER BY count DESC
            """,
            params,
        )

    def class_distribution(self, mode: str | None = None) -> pd.DataFrame:
        where = "WHERE mode = ?" if mode else ""
        params: tuple = (mode,) if mode else ()
        return self.query(
            f"SELECT prediction, COUNT(*) AS count FROM predictions {where} "
            f"GROUP BY prediction ORDER BY count DESC",
            params,
        )

    def timeline(self, bucket: str = "minute", mode: str | None = None) -> pd.DataFrame:
        fmt = {"minute": "%Y-%m-%dT%H:%M", "hour": "%Y-%m-%dT%H", "day": "%Y-%m-%d"}.get(
            bucket, "%Y-%m-%dT%H:%M"
        )
        where = "WHERE mode = ?" if mode else ""
        params: tuple = (mode,) if mode else ()
        return self.query(
            f"""
            SELECT strftime('{fmt}', timestamp) AS bucket,
                   COUNT(*) AS total,
                   SUM(CASE WHEN is_attack = 1 THEN 1 ELSE 0 END) AS attacks,
                   AVG(risk_score) AS avg_risk
              FROM predictions {where}
             GROUP BY bucket
             ORDER BY bucket
            """,
            params,
        )

    def protocol_distribution(self, mode: str | None = None) -> pd.DataFrame:
        where = "WHERE mode = ?" if mode else ""
        params: tuple = (mode,) if mode else ()
        return self.query(
            f"""
            SELECT COALESCE(NULLIF(protocol_name, ''), 'OTHER') AS protocol,
                   COUNT(*) AS flows,
                   SUM(packet_count) AS packets,
                   SUM(byte_count) AS bytes,
                   AVG(duration_seconds) AS avg_duration
              FROM flows {where}
             GROUP BY protocol
             ORDER BY flows DESC
            """,
            params,
        )

    def top_talkers(self, limit: int = 10, mode: str | None = None) -> pd.DataFrame:
        where = "WHERE mode = ?" if mode else ""
        params: list[Any] = [mode] if mode else []
        params.append(int(limit))
        return self.query(
            f"""
            SELECT src_ip, COUNT(*) AS flows, SUM(byte_count) AS bytes,
                   SUM(packet_count) AS packets
              FROM flows {where}
             GROUP BY src_ip
             ORDER BY bytes DESC
             LIMIT ?
            """,
            params,
        )

    def recent_events(self, limit: int = 100) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM system_events ORDER BY id DESC LIMIT ?", (int(limit),)
        )

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for table in TABLES:
            frame = self.query(f"SELECT COUNT(*) AS n FROM {table}")
            result[table] = int(frame.iloc[0]["n"]) if not frame.empty else 0
        return result

    # ---------------------------------------------------------- maintenance
    def prune(self, retention_days: int = 30) -> int:
        """Delete rows older than ``retention_days``. ``0`` disables pruning."""
        if not retention_days or int(retention_days) <= 0:
            return 0
        cutoff = f"-{int(retention_days)} days"
        removed = 0
        with self._write() as cursor:
            for table in ("alerts", "predictions", "flows", "system_events", "behaviour_events"):
                cursor.execute(
                    f"DELETE FROM {table} WHERE timestamp < datetime('now', ?)", (cutoff,)
                )
                removed += cursor.rowcount if cursor.rowcount > 0 else 0
        if removed:
            logger.info("Pruned %d row(s) older than %d day(s)", removed, retention_days)
        return removed

    def clear(self, tables: Sequence[str] | None = None) -> None:
        """Delete every row from the given tables (default: all data tables)."""
        with self._write() as cursor:
            for table in tables or TABLES:
                cursor.execute(f"DELETE FROM {table}")
        logger.info("Cleared tables: %s", ", ".join(tables or TABLES))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<Database path={self.path}>"


# ------------------------------------------------------------------ helpers
def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number


def _json_number(value: Any) -> float | None:
    """A finite number, or ``None`` (JSON null) for NaN and infinities."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number or number in (float("inf"), float("-inf")) else number


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _load_json(raw: Any, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _make_uid(identity: Mapping[str, Any]) -> str:
    return (
        f"{identity.get('src_ip','')}:{identity.get('src_port','')}-"
        f"{identity.get('dst_ip','')}:{identity.get('dst_port','')}-"
        f"{identity.get('protocol','')}-{identity.get('start_time','')}"
    )


def get_database(settings=None) -> Database:
    """Build a :class:`Database` from the application settings."""
    from src.config.settings import get_settings

    settings = settings or get_settings()
    cfg = settings.section("database")
    return Database(
        settings.path("database_path"),
        journal_mode=str(cfg.get("journal_mode", "WAL")),
        timeout=float(cfg.get("timeout", 30.0)),
    )


__all__ = ["Database", "get_database"]
