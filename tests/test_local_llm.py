"""The local AI review: plain-code checks, reply validation and the local-only rule."""

from __future__ import annotations

import json

import pytest

from src.utils.errors import ExplainerError
from src.xai.local_llm import (
    FALSE_ALARM,
    NEEDS_REVIEW,
    OllamaClient,
    build_facts,
    contradictions,
    decide_verdict,
    flow_checks,
    parse_review,
    review_prediction,
)

SSDP_REPLY = {
    "src_ip": "192.168.1.128",
    "src_port": 1900,
    "dst_ip": "192.168.1.35",
    "dst_port": 55342,
    "protocol_name": "UDP",
    "packet_count": 4,
    "byte_count": 1352,
    "duration_seconds": 3.0,
}
PAYLOAD = {
    "prediction": "DoS Slowhttptest",
    "confidence": 0.7036,
    "risk_score": 61.1,
    "top_features": [
        {"label": "Fwd Packet Length Mean", "value": 338.0, "contribution": 2.05},
    ],
}


def _reply(summary="Discovery traffic.", **extra):
    return json.dumps(
        {"summary": summary, "reasons": ["UDP"], "next_steps": ["Check it"], **extra}
    )


def test_tcp_only_label_on_udp_is_ruled_out():
    assert contradictions("DoS Slowhttptest", SSDP_REPLY)


def test_tcp_only_label_on_tcp_is_not_ruled_out():
    assert not contradictions("DoS Slowhttptest", {**SSDP_REPLY, "protocol_name": "TCP"})


def test_benign_label_is_not_ruled_out():
    assert not contradictions("BENIGN", SSDP_REPLY)


def test_missing_protocol_rules_nothing_out():
    assert not contradictions("DoS Slowhttptest", {**SSDP_REPLY, "protocol_name": None})


def test_checks_name_the_service():
    assert "SSDP" in " ".join(flow_checks("DoS Slowhttptest", SSDP_REPLY))


def test_private_addresses_alone_do_not_clear_an_attack():
    record = {**SSDP_REPLY, "protocol_name": "TCP", "src_port": 49152, "dst_port": 22}
    assert flow_checks("SSH-Patator", record) == []
    assert decide_verdict("SSH-Patator", record) == NEEDS_REVIEW


def test_only_a_contradiction_gives_a_false_alarm_verdict():
    assert decide_verdict("DoS Slowhttptest", SSDP_REPLY) == FALSE_ALARM
    assert decide_verdict("PortScan", {**SSDP_REPLY, "protocol_name": "TCP"}) == NEEDS_REVIEW


def test_multicast_destination_is_noted():
    checks = flow_checks("BENIGN", {"dst_ip": "224.0.0.251", "src_ip": "192.168.1.128"})
    assert any("multicast" in check for check in checks)


def test_facts_include_the_checks_and_the_flow():
    facts = build_facts(SSDP_REPLY, PAYLOAD)
    assert facts["flow"]["protocol"] == "UDP"
    assert facts["checks"]
    assert facts["verdict"] == FALSE_ALARM
    assert facts["top_features"][0]["pushed_towards_label"] is True


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "[]",
        json.dumps({"summary": ""}),
    ],
)
def test_malformed_reply_is_an_error(reply):
    with pytest.raises(ExplainerError):
        parse_review(reply, "m", NEEDS_REVIEW, [])


def test_reassuring_text_on_an_open_alert_gets_a_warning():
    reply = _reply(summary="This looks like a small, benign request.")
    assert parse_review(reply, "m", NEEDS_REVIEW, []).text_warning


def test_cleared_alert_may_use_reassuring_text():
    reply = _reply(summary="This is normal discovery traffic.")
    assert not parse_review(reply, "m", FALSE_ALARM, []).text_warning


def test_model_cannot_change_the_verdict():
    reply = _reply(verdict=FALSE_ALARM)
    assert parse_review(reply, "m", NEEDS_REVIEW, []).verdict == NEEDS_REVIEW


@pytest.mark.parametrize("url", ["http://example.com:11434", "http://192.168.1.128:11434"])
def test_flow_data_cannot_be_sent_to_another_host(url):
    with pytest.raises(ExplainerError):
        OllamaClient(url)


def test_unreachable_server_gives_a_clear_error():
    client = OllamaClient("http://127.0.0.1:9", timeout=1.0)
    with pytest.raises(ExplainerError, match="not reachable"):
        client.list_models()


def test_review_passes_facts_to_the_model_and_returns_the_checks():
    class FakeClient:
        def chat_json(self, model, system, user, *, keep_alive):
            self.user = json.loads(user)
            return _reply()

    client = FakeClient()
    review = review_prediction(SSDP_REPLY, PAYLOAD, client=client, model="m")
    assert client.user["flow"]["source"] == "192.168.1.128:1900"
    assert review.verdict_label == "Likely false alarm"
    assert review.checks == client.user["checks"]
