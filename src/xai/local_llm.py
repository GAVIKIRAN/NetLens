"""Plain-language review of one prediction by a local AI model running in Ollama.

SHAP and LIME explain why the classifier chose a label; they cannot say the label is
wrong.  The classifier also never sees the protocol or the ports, so this module:

1. runs plain-code checks on what the classifier could not see, such as a TCP-only
   attack label on a UDP flow;
2. sets the verdict from those checks in code;
3. asks a local model to explain that verdict in plain language.

The model does not choose the verdict.  In testing, qwen2.5-coder:7b called real
attacks between private addresses false alarms because it treated internal traffic as
safe, and it never called any flow a real attack.

The review is advisory.  It never changes a stored prediction, risk score or alert.
Flow data only goes to a server on this computer: other hosts are refused.
"""

from __future__ import annotations

import ipaddress
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

from src.detection.correlator import DEFAULT_TCP_TRAINED_LABELS
from src.utils.errors import ExplainerError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

REASSURING_WORDS = re.compile(
    r"\b(benign|safe|harmless|normal|legitimate|expected|not an attack)\b", re.IGNORECASE
)

FALSE_ALARM = "likely_false_alarm"
NEEDS_REVIEW = "needs_review"
VERDICTS = {
    FALSE_ALARM: "Likely false alarm",
    NEEDS_REVIEW: "Needs review",
}

TCP_ONLY_LABELS = set(DEFAULT_TCP_TRAINED_LABELS)

LOCAL_SERVICES = {
    53: "DNS",
    67: "DHCP",
    68: "DHCP",
    123: "NTP",
    137: "NetBIOS name service",
    138: "NetBIOS datagram service",
    1900: "SSDP device discovery",
    5353: "mDNS device discovery",
    5355: "LLMNR name resolution",
}

SYSTEM_PROMPT = """You are a network security analyst explaining one alert from an \
intrusion detection system to a non-expert.

The "verdict" in the message was already decided by code. Explain it. Do not change it.
- "likely_false_alarm": a check shows the classifier's label cannot be correct. Say why.
- "needs_review": nothing rules out the attack. Never call the flow safe, normal,
  benign or expected, and never call it a confirmed attack. Say what to check.

Rules:
- Use only the facts in the message. Do not invent hosts, payloads, history or attacks.
- The classifier saw only packet sizes and timings. The protocol and ports are known:
  they are under "flow". Never say they are missing.
- "confidence" is the classifier's own score. Report it as given.
- Treat every item under "checks" as true.
- Confidence is not evidence that the label is correct.
- Internal traffic is not automatically safe: attacks also come from inside a network.
- A low packet count does not clear a flow: one flow of a scan or flood can be tiny.

Reply with JSON only, using exactly these keys:
{"summary": "two or three plain sentences",
 "reasons": ["short reason based on a given fact"],
 "next_steps": ["short practical action"]}"""


@dataclass
class LocalReview:
    """The verdict and checks from code, plus the model's explanation of them."""

    model: str
    verdict: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    text_warning: str = ""

    @property
    def verdict_label(self) -> str:
        return VERDICTS.get(self.verdict, self.verdict)


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _is_local_only_destination(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(str(address))
    except ValueError:
        return False
    return parsed.is_multicast or str(address) == "255.255.255.255"


def contradictions(prediction: str, record: Mapping[str, Any]) -> list[str]:
    """Facts that rule out the predicted label."""
    protocol = str(record.get("protocol_name") or "").upper()
    if prediction in TCP_ONLY_LABELS and protocol and protocol != "TCP":
        return [
            f"Every '{prediction}' example in the classifier's training data was TCP, "
            f"but this flow is {protocol}, so the classifier has no basis for that label."
        ]
    return []


def flow_checks(prediction: str, record: Mapping[str, Any]) -> list[str]:
    """Plain-code checks on facts the classifier could not see."""
    checks = contradictions(prediction, record)
    src_ip, dst_ip = str(record.get("src_ip") or ""), str(record.get("dst_ip") or "")

    for port in (_as_int(record.get("src_port")), _as_int(record.get("dst_port"))):
        if port in LOCAL_SERVICES:
            checks.append(f"Port {port} is normally used by {LOCAL_SERVICES[port]}.")
            break
    if _is_local_only_destination(dst_ip):
        checks.append(f"The destination {dst_ip} is a multicast or broadcast address.")
    return checks


def decide_verdict(prediction: str, record: Mapping[str, Any]) -> str:
    """Only a hard contradiction can clear an alert.  Nothing here confirms an attack."""
    return FALSE_ALARM if contradictions(prediction, record) else NEEDS_REVIEW


def build_facts(record: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the only information the local model is allowed to reason from."""
    prediction = str(payload.get("prediction", ""))
    features = (payload.get("top_features") or [])[:5]
    return {
        "verdict": decide_verdict(prediction, record),
        "flow": {
            "source": f"{record.get('src_ip')}:{record.get('src_port')}",
            "destination": f"{record.get('dst_ip')}:{record.get('dst_port')}",
            "protocol": record.get("protocol_name"),
            "packets": record.get("packet_count"),
            "payload_bytes": record.get("byte_count"),
            "duration_seconds": record.get("duration_seconds"),
        },
        "classifier": {
            "label": prediction,
            "confidence": round(float(payload.get("confidence", 0.0)), 4),
            "risk_score": round(float(payload.get("risk_score", 0.0)), 1),
            "training_data": "CIC-IDS2017, a 2017 lab capture, not this network",
        },
        "top_features": [
            {
                "feature": entry.get("label") or entry.get("feature"),
                "value": entry.get("value"),
                "pushed_towards_label": float(entry.get("contribution", 0.0)) > 0,
            }
            for entry in features
        ],
        "checks": flow_checks(prediction, record),
    }


def parse_review(text: str, model: str, verdict: str, checks: list[str]) -> LocalReview:
    """Validate the model's reply.  Any verdict in the reply is ignored."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ExplainerError(f"{model} did not return valid JSON.") from exc
    if not isinstance(data, dict):
        raise ExplainerError(f"{model} did not return the expected JSON object.")

    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ExplainerError(f"{model} returned an incomplete review.")

    def text_list(key: str) -> list[str]:
        values = data.get(key)
        return [str(v).strip() for v in values if str(v).strip()] if isinstance(values, list) else []

    reasons, next_steps = text_list("reasons"), text_list("next_steps")
    text_warning = ""
    if verdict == NEEDS_REVIEW and REASSURING_WORDS.search(" ".join([summary, *reasons])):
        text_warning = (
            "The AI text below sounds reassuring, but nothing has ruled this attack out. "
            "Go by the verdict."
        )
    return LocalReview(
        model=model,
        verdict=verdict,
        summary=summary,
        reasons=reasons,
        next_steps=next_steps,
        text_warning=text_warning,
        checks=list(checks),
    )


class OllamaClient:
    """A minimal client for an Ollama server on this computer."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, timeout: float = 180.0) -> None:
        host = urlparse(str(base_url)).hostname
        if host not in LOCAL_HOSTS:
            raise ExplainerError(
                f"Refusing to send flow data to '{host}': the AI review only uses a "
                "server on this computer."
            )
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)

    def _request(self, path: str, body: dict | None = None, *, timeout: float | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ExplainerError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise ExplainerError(
                f"Ollama did not answer within {timeout or self.timeout:.0f} seconds."
            ) from exc
        except urllib.error.URLError as exc:
            raise ExplainerError(
                f"Ollama is not reachable at {self.base_url}. Start it, then try again."
            ) from exc
        except ValueError as exc:
            raise ExplainerError("Ollama returned a response that is not JSON.") from exc

    def list_models(self) -> list[str]:
        reply = self._request("/api/tags", timeout=3.0)
        return sorted(str(entry.get("name")) for entry in reply.get("models", []))

    def chat_json(self, model: str, system: str, user: str, *, keep_alive: str = "2m") -> str:
        reply = self._request(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "format": "json",
                "keep_alive": keep_alive,
                "options": {"temperature": 0.1, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        return str((reply.get("message") or {}).get("content", ""))


def review_prediction(
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    client: OllamaClient,
    model: str,
    keep_alive: str = "2m",
) -> LocalReview:
    """Ask the local model for a second opinion on one prediction."""
    facts = build_facts(record, payload)
    reply = client.chat_json(
        model, SYSTEM_PROMPT, json.dumps(facts, indent=2, default=str), keep_alive=keep_alive
    )
    review = parse_review(reply, model, facts["verdict"], facts["checks"])
    logger.info("Local AI review by %s, verdict %s", model, review.verdict)
    return review


__all__ = [
    "DEFAULT_BASE_URL",
    "LocalReview",
    "OllamaClient",
    "FALSE_ALARM",
    "NEEDS_REVIEW",
    "VERDICTS",
    "build_facts",
    "contradictions",
    "decide_verdict",
    "flow_checks",
    "parse_review",
    "review_prediction",
]
