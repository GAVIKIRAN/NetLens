"""Monitoring orchestrator - the component the dashboard actually drives.

It owns one processing path that every mode feeds into::

    packet -> behaviour detector (layer 2)
           -> flow manager -> flow record -> preprocessing -> model (layer 1)
           -> correlator: final decision and risk (layer 3)
           -> alert, (optional inline SHAP), SQLite, ring buffer (layer 4)

Scoring and database writes run on a worker thread behind a bounded queue, so a slow
disk or a SHAP computation cannot stall packet capture.

Three modes share that path:

``LIVE``
    Scapy captures packets from a real interface, the flow manager assembles
    them, and completed flows are classified.  Requires Npcap + Administrator
    on Windows.

``DEMO``
    Pre-recorded / synthetic flow records are replayed at a configurable rate.
    No packets are captured or transmitted; this is the safe presentation mode.

``PCAP``
    A saved capture file is replayed through the real flow manager offline.

The service is thread-safe: capture runs on background threads and the
dashboard reads counters and the ring buffer from the Streamlit thread.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Mapping

from src.config.settings import Settings, get_settings
from src.database.database import Database, get_database
from src.detection.behavior import BehaviorDetector, Finding
from src.detection.correlator import BENIGN, Correlator
from src.inference.predictor import Predictor, PredictionResult
from src.inference.risk import RiskScorer
from src.network.feature_extractor import flow_to_features
from src.network.flow_manager import Flow, FlowManager
from src.network.interfaces import capture_driver_status
from src.network.packet_capture import PacketCapture, read_pcap
from src.simulation.replay import DemoReplaySource
from src.utils.errors import CaptureError, DatasetError, ModelError, NIDSError
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODE_LIVE = "LIVE"
MODE_DEMO = "DEMO"
MODE_PCAP = "PCAP"
MODES = (MODE_LIVE, MODE_DEMO, MODE_PCAP)


@dataclass
class MonitorStats:
    """Counters exposed to the dashboard."""

    started_at: float = 0.0
    flows_processed: int = 0
    attacks_detected: int = 0
    alerts_raised: int = 0
    packets_seen: int = 0
    errors: int = 0
    last_flow_at: float = 0.0
    last_error: str = ""
    explanations_computed: int = 0
    final_attacks: int = 0
    final_suspicious: int = 0
    behaviour_findings: int = 0
    records_dropped: int = 0

    def uptime(self) -> float:
        return max(0.0, time.time() - self.started_at) if self.started_at else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "flows_processed": self.flows_processed,
            "attacks_detected": self.attacks_detected,
            "alerts_raised": self.alerts_raised,
            "packets_seen": self.packets_seen,
            "errors": self.errors,
            "last_error": self.last_error,
            "explanations_computed": self.explanations_computed,
            "final_attacks": self.final_attacks,
            "final_suspicious": self.final_suspicious,
            "behaviour_findings": self.behaviour_findings,
            "records_dropped": self.records_dropped,
            "uptime_seconds": round(self.uptime(), 1),
            "attack_rate": (
                round(100.0 * self.attacks_detected / self.flows_processed, 2)
                if self.flows_processed
                else 0.0
            ),
        }


class MonitorService:
    """Single point of control for capture, classification and persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        database: Database | None = None,
        predictor: Predictor | None = None,
        explanation_service: Any | None = None,
        buffer_size: int = 500,
    ) -> None:
        self.settings = settings or get_settings()
        self.database = database or get_database(self.settings)
        self._predictor = predictor
        self._explainer = explanation_service
        self.risk_scorer = RiskScorer.from_settings(self.settings)

        self.mode: str = ""
        self.stats = MonitorStats()
        self.recent: Deque[dict[str, Any]] = deque(maxlen=int(buffer_size))
        self.recent_findings: Deque[dict[str, Any]] = deque(maxlen=100)
        self.errors: list[str] = []

        self._lock = threading.RLock()
        self._running = threading.Event()
        self._threads: list[threading.Thread] = []
        self._capture: PacketCapture | None = None
        self._flow_manager: FlowManager | None = None
        self._demo_source: DemoReplaySource | None = None

        self.alert_min_band = str(self.settings.get("risk.alert_min_band", "MEDIUM"))
        self.benign_label = str(self.settings.get("inference.benign_label", "BENIGN"))

        detection_cfg = self.settings.section("detection")
        self.detection_enabled = bool(detection_cfg.get("enabled", True))
        self._behavior_config = detection_cfg.get("behavior") or {}
        self.correlator = Correlator(
            self.risk_scorer,
            detection_cfg.get("correlation") or {},
            alert_min_band=self.alert_min_band,
        )
        self._behavior: BehaviorDetector | None = None
        self._queue: queue.Queue = queue.Queue(
            maxsize=int(detection_cfg.get("worker_queue_size", 5000))
        )
        self._worker_thread: threading.Thread | None = None
        self._worker_stop = threading.Event()

    # ------------------------------------------------------------ lazy load
    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            self._predictor = Predictor.load(self.settings)
        return self._predictor

    @property
    def explainer(self):
        if self._explainer is None:
            from src.xai.explanation_engine import ExplanationService

            self._explainer = ExplanationService(self.predictor, self.settings)
        return self._explainer

    def model_ready(self) -> bool:
        return Predictor.artefacts_exist(self.settings)

    # ---------------------------------------------------------------- start
    def start(
        self,
        mode: str = MODE_DEMO,
        *,
        interface: str | None = None,
        pcap_path: str | Path | None = None,
        replay_rate: float | None = None,
    ) -> None:
        """Start monitoring in the requested mode."""
        mode = str(mode).upper()
        if mode not in MODES:
            raise NIDSError(f"Unknown monitoring mode '{mode}'. Use one of {MODES}.")
        if self.is_running:
            logger.warning("Monitor already running in %s mode; stop it first.", self.mode)
            return
        if not self.model_ready():
            raise ModelError(
                "No trained model found, so monitoring cannot start.",
                hint="Run `python scripts/train_model.py` first.",
            )

        _ = self.predictor  # fail fast if the artefacts are inconsistent
        with self._lock:
            self.mode = mode
            self.stats = MonitorStats(started_at=time.time())
            self.recent.clear()
            self.recent_findings.clear()
            self.errors.clear()
            self._threads.clear()
            self._running.set()

        self.database.start_session(mode)
        self.database.log_event("INFO", "monitor", f"Monitoring started in {mode} mode")
        try:
            if mode == MODE_LIVE:
                self._start_live(interface)
            elif mode == MODE_DEMO:
                self._start_demo(replay_rate)
            else:
                self._start_pcap(pcap_path)
        except Exception:
            self._running.clear()
            self.mode = ""
            try:
                self.database.end_session(self.stats.to_dict())
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error while closing the session after a failed start")
            raise
        logger.info("Monitoring started in %s mode", mode)

    def _start_live(self, interface: str | None) -> None:
        # Fail here with the real reason, instead of appearing to start and then
        # reporting a vague capture error from the sniffing thread.
        driver_ok, driver_message = capture_driver_status()
        if not driver_ok:
            raise CaptureError(
                "Live capture needs the packet-capture driver, which is not installed.",
                hint=f"{driver_message}. DEMO and PCAP modes work without it.",
            )
        network_cfg = self.settings.section("network")
        self._prepare_packet_path()
        self._start_worker()
        self._capture = PacketCapture(
            on_packet=self._on_packet,
            interface=interface if interface is not None else network_cfg.get("interface"),
            bpf_filter=str(network_cfg.get("bpf_filter", "ip") or ""),
            duration=float(network_cfg.get("capture_duration", 0) or 0),
            on_error=self._on_error,
        )
        self._capture.start()
        self._spawn(self._sweeper, "nids-sweeper")

    def _start_demo(self, replay_rate: float | None) -> None:
        demo_cfg = self.settings.section("demo")
        self._demo_source = DemoReplaySource(
            demo_cfg.get("source_file"),
            rate=float(replay_rate or demo_cfg.get("replay_rate", 4.0)),
            loop=bool(demo_cfg.get("loop", True)),
            seed=int(self.settings.get("training.random_state", 42)),
        )
        self._spawn(self._demo_loop, "nids-demo")

    def _start_pcap(self, pcap_path: str | Path | None) -> None:
        if not pcap_path:
            raise CaptureError("PCAP mode requires a capture file path.")
        self._prepare_packet_path()
        self._start_worker()
        path = Path(pcap_path)
        self._spawn(lambda: self._pcap_loop(path), "nids-pcap")

    def _prepare_packet_path(self) -> None:
        """Build the flow manager and the behaviour detector for a packet source."""
        network_cfg = self.settings.section("network")
        self._flow_manager = FlowManager(
            flow_timeout=float(network_cfg.get("flow_timeout", 120.0)),
            activity_timeout=float(network_cfg.get("activity_timeout", 5.0)),
            subflow_timeout=float(network_cfg.get("subflow_timeout", 1.0)),
            max_active_flows=int(network_cfg.get("max_active_flows", 20000)),
            on_flow_complete=self._on_flow_complete,
            min_packets_to_emit=int(network_cfg.get("min_packets_per_flow", 2)),
        )
        self._behavior = (
            BehaviorDetector(self._behavior_config, on_finding=self._on_finding)
            if self.detection_enabled
            else None
        )

    def _start_worker(self) -> None:
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._guard(self._worker_loop), name="nids-worker", daemon=True
        )
        self._worker_thread.start()

    def _stop_worker(self, timeout: float = 30.0) -> None:
        """Let the worker finish what is queued, then stop it."""
        self._worker_stop.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("The worker did not finish within %.0f s", timeout)
        self._worker_thread = None

    def _worker_loop(self) -> None:
        while True:
            try:
                kind, *item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._worker_stop.is_set():
                    return
                continue
            try:
                if kind == "flow":
                    self.process_record(item[0], item[1])
                else:
                    self._store_finding(item[0])
            except Exception as exc:  # the worker must outlive one bad record
                self._on_error(exc)

    def _submit(self, *item) -> None:
        """Hand work to the worker, or do it here when no worker is running."""
        if self._worker_thread is None:
            if item[0] == "flow":
                self.process_record(item[1], item[2])
            else:
                self._store_finding(item[1])
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.stats.records_dropped += 1
                dropped = self.stats.records_dropped
            if dropped == 1 or dropped % 1000 == 0:
                logger.warning("Worker queue full: %d record(s) dropped so far", dropped)

    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=self._guard(target), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _guard(self, target):
        def _wrapped() -> None:
            try:
                target()
            except Exception as exc:  # pragma: no cover - defensive
                self._on_error(exc)
        return _wrapped

    # ----------------------------------------------------------------- stop
    def stop(self) -> None:
        """Stop every worker and flush any half-assembled flows."""
        if not self._running.is_set():
            return
        self._running.clear()
        if self._demo_source is not None:
            self._demo_source.stop()
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error while stopping the capture")
        if self._flow_manager is not None:
            try:
                self._flow_manager.flush()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error while flushing flows")
        if self._behavior is not None:
            self._behavior.tick()
        self._stop_worker()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=3.0)
        self._threads.clear()
        try:
            self.database.end_session(self.stats.to_dict())
        except Exception:  # pragma: no cover - defensive
            logger.exception("Error while closing the session")
        self.database.log_event("INFO", "monitor", f"Monitoring stopped ({self.mode})")
        logger.info("Monitoring stopped (%s)", self.mode or "idle")
        self.mode = ""

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # -------------------------------------------------------------- workers
    def _sweeper(self) -> None:
        interval = float(self.settings.get("network.sweep_interval", 2.0))
        while self._running.is_set():
            time.sleep(interval)
            if self._flow_manager is not None:
                try:
                    self._flow_manager.expire()
                except Exception as exc:  # pragma: no cover - defensive
                    self._on_error(exc)
            if self._behavior is not None:
                try:
                    self._behavior.tick(time.time())
                except Exception as exc:  # pragma: no cover - defensive
                    self._on_error(exc)

    def _demo_loop(self) -> None:
        source = self._demo_source
        if source is None:
            return
        for record in source.stream():
            if not self._running.is_set():
                break
            self.process_record(
                record.features, record.identity, ground_truth=record.ground_truth
            )

    def _pcap_loop(self, path: Path) -> None:
        try:
            read_pcap(path, self._on_packet)
        finally:
            self._finish_offline()
            self._stop_worker()
            self._running.clear()

    def _finish_offline(self) -> None:
        """End a finite packet source: close open flows and age unanswered probes."""
        if self._flow_manager is not None:
            self._flow_manager.flush()
        if self._behavior is not None:
            detector = self._behavior
            detector.tick(detector._clock + detector.unanswered_after + 1.0)

    def process_packets(self, packets) -> None:
        """Run packets through every layer on this thread (evaluation and tests)."""
        self.mode = self.mode or MODE_PCAP
        self._prepare_packet_path()
        for packet in packets:
            self._on_packet(packet)
        self._finish_offline()

    # ------------------------------------------------------------ callbacks
    def _on_packet(self, packet) -> None:
        with self._lock:
            self.stats.packets_seen += 1
        # Before flow assembly: a lone SYN never becomes a flow, but is a scan probe.
        if self._behavior is not None:
            try:
                self._behavior.observe(packet)
            except Exception as exc:  # detection must never stop capture
                self._on_error(exc)
        if self._flow_manager is not None:
            self._flow_manager.add_packet(packet)

    def _on_flow_complete(self, flow: Flow) -> None:
        try:
            features = flow_to_features(flow)
        except Exception as exc:  # pragma: no cover - defensive
            self._on_error(exc)
            return
        self._submit("flow", features, flow.identity())

    def _on_finding(self, finding: Finding) -> None:
        self._submit("finding", finding)

    def _store_finding(self, finding: Finding) -> None:
        """Layer 4 for a behavioural finding: event row, ring buffer and alert."""
        score, formula = self.correlator.behaviour_risk(finding)
        level = self.risk_scorer.band(score)
        mode = self.mode or MODE_LIVE
        event = {
            "kind": finding.kind,
            "decision": finding.decision,
            "src_ip": finding.src_ip,
            "dst_ip": finding.dst_ip,
            "dst_port": finding.dst_port,
            "risk_score": score,
            "risk_level": level,
            "risk_formula": formula,
            "reason": finding.reason,
            "indicators": finding.indicators,
            "repeats": finding.repeats,
        }
        logger.warning(
            "Behavioural detection [%s, risk %.0f %s]: %s",
            finding.decision, score, level, finding.reason,
        )
        raise_alert = RiskScorer.at_least(level, self.alert_min_band)
        try:
            self.database.insert_behaviour_event(event, mode=mode)
            if raise_alert:
                self.database.insert_alert(
                    None,
                    None,
                    {
                        "src_ip": finding.src_ip,
                        "dst_ip": finding.dst_ip or "multiple",
                        "src_port": 0,
                        "dst_port": finding.dst_port,
                        "protocol_name": "TCP",
                    },
                    {
                        "risk_level": level,
                        "prediction": f"{finding.label} (behaviour)",
                        "confidence": 0.0,
                        "risk_score": score,
                        "explanation": f"{finding.reason}\n\nRisk: {formula}",
                        "top_features": [],
                    },
                    mode=mode,
                )
        except NIDSError as exc:
            self._on_error(exc)
        with self._lock:
            self.stats.behaviour_findings += 1
            if raise_alert:
                self.stats.alerts_raised += 1
            self.recent_findings.appendleft(
                {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "source": finding.src_ip,
                    "target": (
                        f"{finding.dst_ip or 'multiple hosts'}"
                        + (f":{finding.dst_port}" if finding.dst_port else "")
                    ),
                    "detection": finding.label,
                    "decision": finding.decision,
                    "risk_score": score,
                    "risk_level": level,
                    "reason": finding.reason,
                }
            )

    def _on_error(self, error: Exception) -> None:
        message = str(error)
        with self._lock:
            self.stats.errors += 1
            self.stats.last_error = message
            self.errors.append(message)
            self.errors = self.errors[-25:]
        logger.error("Monitor error: %s", message)
        try:
            self.database.log_event("ERROR", "monitor", message)
        except Exception:  # pragma: no cover - defensive
            pass

    # -------------------------------------------------- the processing path
    def process_record(
        self,
        features: Mapping[str, float],
        identity: Mapping[str, Any],
        *,
        ground_truth: str | None = None,
    ) -> PredictionResult | None:
        """Classify one flow record and persist everything about it."""
        try:
            result = self.predictor.predict_one(dict(features))
        except NIDSError as exc:
            self._on_error(exc)
            return None
        except Exception as exc:  # pragma: no cover - unexpected
            self._on_error(exc)
            return None

        payload = result.to_dict()
        payload["formula"] = result.risk.formula if result.risk else ""
        top_features: list[dict[str, Any]] = []
        explanation_text = ""
        explainer_name = ""

        # Layer 2 result for this flow, and layer 3: the model's label stays as it is.
        findings: list[Finding] = []
        if self._behavior is not None:
            try:
                findings = self._behavior.findings_for(
                    str(identity.get("src_ip", "")),
                    str(identity.get("dst_ip", "")),
                    int(identity.get("dst_port") or 0),
                    at=identity.get("end_time"),
                )
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Behaviour lookup skipped: %s", exc)
        decision = self.correlator.decide(
            ml_label=result.predicted_label,
            ml_is_attack=result.is_attack,
            confidence=result.confidence,
            ml_risk_score=result.risk.score if result.risk else 0.0,
            ml_risk_formula=result.risk.formula if result.risk else "",
            protocol_name=identity.get("protocol_name"),
            findings=findings,
        )
        payload.update(decision.to_payload())

        is_alert = decision.raises_flow_alert

        if is_alert:
            try:
                service = self.explainer
                if service.inline_capable:
                    top_features = service.quick_top_features(result)
                    if top_features:
                        from src.xai.explanation_engine import build_explanation

                        narrative = build_explanation(
                            result.predicted_label,
                            result.confidence,
                            top_features,
                            risk_score=decision.risk_score,
                            risk_level=decision.risk_level,
                            benign_label=self.benign_label,
                            explainer=service.shap.explainer_type if service.shap else "SHAP",
                        )
                        explanation_text = narrative.text()
                        explainer_name = (
                            service.shap.explainer_type if service.shap else "SHAP"
                        )
                        with self._lock:
                            self.stats.explanations_computed += 1
            except Exception as exc:
                logger.debug("Inline explanation skipped: %s", exc)

        payload["top_features"] = top_features
        payload["explanation"] = explanation_text
        payload["explainer"] = explainer_name

        flow_id = prediction_id = None
        try:
            flow_id = self.database.insert_flow(
                identity, features, mode=self.mode or MODE_DEMO
            )
            prediction_id = self.database.insert_prediction(
                flow_id, payload, mode=self.mode or MODE_DEMO, ground_truth=ground_truth
            )
            if is_alert:
                self.database.insert_alert(
                    prediction_id, flow_id, identity, payload, mode=self.mode or MODE_DEMO
                )
        except NIDSError as exc:
            self._on_error(exc)

        with self._lock:
            self.stats.flows_processed += 1
            self.stats.last_flow_at = time.time()
            if result.is_attack:
                self.stats.attacks_detected += 1
            if decision.final == BENIGN:
                pass
            elif decision.final == "ATTACK":
                self.stats.final_attacks += 1
            else:
                self.stats.final_suspicious += 1
            if is_alert:
                self.stats.alerts_raised += 1
            self.recent.appendleft(
                {
                    "prediction_id": prediction_id,
                    "flow_id": flow_id,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "src": f"{identity.get('src_ip','')}:{identity.get('src_port','')}",
                    "dst": f"{identity.get('dst_ip','')}:{identity.get('dst_port','')}",
                    "protocol": identity.get("protocol_name", ""),
                    "prediction": result.predicted_label,
                    "confidence": round(float(result.confidence), 4),
                    "behaviour": decision.behaviour_result,
                    "final": decision.final,
                    "reason": decision.reason,
                    "ml_family": decision.ml_family,
                    "risk_score": round(float(decision.risk_score), 1),
                    "risk_level": decision.risk_level,
                    "packets": identity.get("packets", 0),
                    "bytes": identity.get("bytes", 0),
                    "ground_truth": ground_truth or "",
                    "is_attack": result.is_attack,
                }
            )
        return result

    # --------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        """Everything the dashboard header needs, in one thread-safe read."""
        with self._lock:
            payload: dict[str, Any] = {
                "running": self.is_running,
                "mode": self.mode or "IDLE",
                "model_ready": self.model_ready(),
                "stats": self.stats.to_dict(),
                "errors": list(self.errors),
                "alert_min_band": self.alert_min_band,
            }
        if self._capture is not None:
            payload["capture"] = self._capture.stats()
        if self._flow_manager is not None:
            payload["flows"] = self._flow_manager.stats()
        if self._demo_source is not None:
            payload["demo"] = {
                "file": str(self._demo_source.path),
                "records": len(self._demo_source),
                "rate": self._demo_source.rate,
                "classes": self._demo_source.class_counts,
            }
        if self._behavior is not None:
            payload["behaviour"] = self._behavior.stats()
        if self._predictor is not None:
            payload["model"] = self._predictor.describe()
        if self._explainer is not None:
            payload["xai"] = self._explainer.status()
        return payload

    def recent_frame(self, limit: int = 200):
        """Ring-buffer contents as a ``DataFrame`` for the live table."""
        import pandas as pd

        with self._lock:
            rows = list(self.recent)[: int(limit)]
        return pd.DataFrame(rows)

    def findings_frame(self, limit: int = 50):
        """Behavioural detections from this session, newest first."""
        import pandas as pd

        with self._lock:
            rows = list(self.recent_findings)[: int(limit)]
        return pd.DataFrame(rows)

    def process_batch(
        self, records: list[tuple[Mapping[str, float], Mapping[str, Any], str]]
    ) -> int:
        """Process a list of ``(features, identity, ground_truth)`` tuples."""
        count = 0
        for features, identity, ground_truth in records:
            if self.process_record(features, identity, ground_truth=ground_truth):
                count += 1
        return count

    def demo_burst(self, count: int = 25) -> int:
        """Classify ``count`` demo records immediately (used by the UI button)."""
        demo_cfg = self.settings.section("demo")
        source = self._demo_source or DemoReplaySource(
            demo_cfg.get("source_file"), loop=True, shuffle=True
        )
        if not self.mode:
            self.mode = MODE_DEMO
            if not self.stats.started_at:
                self.stats.started_at = time.time()
        records = source.take(int(count))
        own_session = not self.is_running and self.database.current_session_id is None
        if own_session:
            self.database.start_session(MODE_DEMO, label=f"Demo burst ({len(records)})")
        try:
            return self.process_batch(
                [(r.features, r.identity, r.ground_truth) for r in records]
            )
        finally:
            if own_session:
                self.database.end_session(self.stats.to_dict())


_SERVICE: MonitorService | None = None


def get_monitor(settings: Settings | None = None) -> MonitorService:
    """Process-wide singleton (Streamlit caches this via ``st.cache_resource``)."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = MonitorService(settings)
    return _SERVICE


__all__ = [
    "MonitorService",
    "MonitorStats",
    "get_monitor",
    "MODE_LIVE",
    "MODE_DEMO",
    "MODE_PCAP",
    "MODES",
]
