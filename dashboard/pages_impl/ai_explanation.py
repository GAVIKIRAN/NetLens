"""AI EXPLANATION - SHAP and LIME for one selected prediction."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from dashboard.components import charts, metrics, tables, ui
from dashboard.state import load_database, load_explainer, load_predictor, load_settings
from dashboard.styles import theme
from src.features.contract import label_for
from src.utils.errors import ExplainerError, NIDSError
from src.xai.local_llm import DEFAULT_BASE_URL, OllamaClient, review_prediction

PAGE_TITLE = "Explain a prediction"


def _rebuild_result(record: dict, predictor):
    """Re-create a PredictionResult from a stored prediction row."""
    stored = record.get("features", {})
    # Null or absent means "not measured". The pipeline imputes NaN as in training,
    # whereas 0 would be a different flow.
    features = {
        name: float("nan") if stored.get(name) is None else float(stored[name])
        for name in predictor.input_features
    }
    return predictor.predict_one(features)


@st.cache_data(show_spinner=False, max_entries=128)
def _explain(prediction_id: int, use_lime: bool, top_k: int, model_name: str,
             *, _record: dict, _predictor, _service) -> dict:
    """SHAP and LIME for one stored prediction, computed once per
    (prediction, settings, model). Without this every click on the page -
    a slider, a toggle - recomputed LIME from scratch. Arguments starting with
    an underscore are not part of the cache key."""
    result = _rebuild_result(_record, _predictor)
    return _service.explain_prediction(result, use_shap=True, use_lime=use_lime, top_k=top_k)


@st.cache_data(show_spinner=False, ttl=30)
def _installed_models(base_url: str) -> tuple[list[str], str]:
    """Ask Ollama for its models at most every 30 s. When Ollama is not running,
    each attempt waits for a connection timeout (about 4 s on Windows), which
    used to be paid on every click on this page."""
    try:
        return OllamaClient(base_url, timeout=3.0).list_models(), ""
    except ExplainerError as exc:
        return [], str(exc)


def _render_local_review(settings, record: dict, payload: dict, prediction_id: int) -> None:
    """Optional second opinion on this prediction from a local Ollama model."""
    config = settings.get("xai.local_llm", {}) or {}
    if not config.get("enabled", False):
        return

    st.subheader("Local AI review")
    if payload.get("prediction") == settings.get("inference.benign_label", "BENIGN"):
        st.info(
            "The current model classifies this flow as benign, so there is no alert to "
            "review. If it's listed as an attack, an older model stored that label."
        )
        return
    st.caption(
        "SHAP and LIME explain why the classifier chose this label, not whether it is "
        "right. This review also checks the protocol and ports, which the classifier "
        "never sees. It runs on this computer and doesn't change the stored prediction."
    )
    client = OllamaClient(
        str(config.get("base_url", DEFAULT_BASE_URL)),
        timeout=float(config.get("timeout_seconds", 180)),
    )
    installed, error = _installed_models(client.base_url)
    if error:
        st.info(f"{error} The rest of this page works without it.")
        return
    if not installed:
        st.info("Ollama is running but has no models installed.")
        return

    default = str(config.get("model", ""))
    choice = st.selectbox(
        "Local model", installed,
        index=installed.index(default) if default in installed else 0,
        key="ax_llm_model",
    )
    key = f"ax_llm_review_{prediction_id}_{choice}"
    if ui.secondary_button("Review this prediction", key="ax_llm_run"):
        with st.spinner(f"Asking {choice}. The first run can take a minute ..."):
            try:
                st.session_state[key] = review_prediction(
                    record, payload, client=client, model=choice,
                    keep_alive=str(config.get("keep_alive", "2m")),
                )
            except ExplainerError as exc:
                st.error(str(exc))

    review = st.session_state.get(key)
    if review is None:
        return
    with st.container(key="card_local_review"):
        st.markdown(f"**{review.verdict_label}** · verdict set by code, not AI")
        if review.checks:
            st.markdown("**Checks**")
            for line in review.checks:
                st.markdown(f"- {line}")
        st.markdown(f"**Explanation by `{review.model}`**")
        if review.text_warning:
            st.warning(review.text_warning)
        st.markdown(review.summary)
        if review.reasons:
            for line in review.reasons:
                st.markdown(f"- {line}")
        if review.next_steps:
            st.markdown("**Suggested next steps**")
            for line in review.next_steps:
                st.markdown(f"- {line}")
        st.caption("The AI text can be wrong. 'Needs review' means nothing ruled the attack out; "
            "it doesn't confirm one.")


def render() -> None:
    settings = load_settings()
    database = load_database()
    predictor = load_predictor()
    service = load_explainer()

    if predictor is None:
        st.error(
            "No trained model is available, so explanations cannot be produced. "
            "Run `python scripts/train_model.py` first."
        )
        return
    if service is None:
        st.error("The explanation service could not be initialised.")
        return

    status = service.status()
    metrics.status_strip(
        [
            (f"SHAP · {status['shap_explainer']}",
             theme.NODE_GREEN if status["shap_available"] else theme.NODE_MAGENTA),
            ("LIME ready" if status["lime_available"] else "LIME unavailable",
             theme.NODE_GREEN if status["lime_available"] else theme.NODE_MAGENTA),
            ("inline explanations on" if status["inline_explanations"]
             else "on-demand explanations only", None),
        ]
    )
    for message in status.get("errors", []):
        st.warning(message)

    # ----------------------------------------------------- pick a prediction
    with st.container(key="card_ax_pick"):
        ui.render(ui.card_header_html("Select a prediction"))
        columns = st.columns([1, 1, 1, 2], vertical_alignment="bottom")
        with columns[0]:
            only_attacks = st.toggle("Attacks only", value=True, key="ax_attacks")
        with columns[1]:
            limit = st.number_input("Search depth", 20, 2000, 200, 20, key="ax_limit")
        with columns[2]:
            use_lime = st.toggle("Compute LIME", value=True, key="ax_lime")
        with columns[3]:
            top_k = st.slider("Features to show", 3, 20,
                              int(settings.get("xai.top_k_features", 8)), key="ax_topk")

        candidates = database.recent_predictions(int(limit), only_attacks=only_attacks)
        if candidates.empty:
            st.info(
                "No stored predictions match. Start monitoring from the Live monitor "
                "page (DEMO mode is safe) and come back."
            )
            return

        # Labels built once - a per-option DataFrame filter cost ~1.5 s per rerun.
        labels = {
            int(row.id): (
                f"#{int(row.id)} · {row.prediction} · "
                f"confidence {float(row.confidence):.0%} · risk {float(row.risk_score):.0f} "
                f"· {row.src_ip} → {row.dst_ip}"
            )
            for row in candidates[["id", "prediction", "confidence", "risk_score", "src_ip", "dst_ip"]]
            .itertuples(index=False)
        }

        def _describe(prediction_id: int) -> str:
            return labels.get(int(prediction_id), f"#{prediction_id}")

        selected = st.selectbox(
            "Prediction", candidates["id"].astype(int).tolist(),
            format_func=_describe, key="ax_prediction",
        )

    record = database.get_prediction(int(selected))
    if record is None:
        st.error("That prediction could not be loaded from the database.")
        return
    if not record.get("features"):
        st.error(
            "The stored flow has no feature vector, so it cannot be re-explained. "
            "This happens for rows written by an older schema version."
        )
        return

    with st.spinner("Computing explanations"):
        try:
            result = _rebuild_result(record, predictor)
            payload = _explain(
                int(selected), bool(use_lime), int(top_k), predictor.model_name,
                _record=record, _predictor=predictor, _service=service,
            )
        except NIDSError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - library specific
            st.error(f"Explanation failed: {exc}")
            return

    # ------------------------------------------------------------- headline
    # Prediction, confidence and risk: three separate figures, three encodings.
    ui.stats([
        {"label": "Prediction", "value": result.predicted_label,
         "badge": ui.attack_badge(result.predicted_label),
         "note": f"stored as {record.get('prediction')}"},
        {"label": "ML confidence", "value": f"{result.confidence:.1%}",
         "note": "probability the model gives the predicted class"},
        {"label": "Risk score", "value": f"{result.risk.score:.0f} / 100",
         "badge": ui.severity_badge(result.risk.level),
         "note": "class severity, confidence and traffic volume"},
    ])

    if result.threshold_applied:
        st.warning(
            f"The model's highest-scoring class was {result.argmax_label} at "
            f"{result.argmax_confidence:.1%}, below the configured decision "
            f"threshold of {predictor.attack_threshold:.0%}, so this flow is "
            "reported as benign."
        )

    facts = st.columns([1, 1])
    with facts[0]:
        decision = [
            ("IDS decision",
             html.escape(str(record.get("final_classification") or result.predicted_label))),
            ("Model said", html.escape(str(record.get("prediction") or ""))),
            ("Behaviour rules", html.escape(str(record.get("behaviour_result") or "none"))),
        ]
        if record.get("detection_reason"):
            decision.append(("Reason", html.escape(str(record["detection_reason"]))))
        if record.get("ground_truth"):
            decision.append(("Demo label", html.escape(str(record["ground_truth"]))))
        with tables.card("ax_decision"):
            ui.render(ui.card_header_html("Decision") + ui.kv_html(decision))
            if record.get("ground_truth"):
                st.caption("The demo label is shown for transparency; the model was not told it")
    with facts[1]:
        with tables.card("ax_formula"):
            ui.render(ui.card_header_html("Risk formula for this flow"))
            st.code(result.risk.formula, language="text")

    # ------------------------------------------------------------ narrative
    narrative = payload.get("narrative") or {}
    with st.container(key="card_narrative"):
        ui.render(ui.card_header_html("Explanation", "plain language"))
        st.markdown(f"**{narrative.get('headline', '')}**")
        st.markdown(narrative.get("summary", ""))
        for line in narrative.get("bullet_points", []):
            st.markdown(f"- {line}")
        st.caption(narrative.get("caveat", ""))

    _render_local_review(settings, record, payload, int(selected))

    # ------------------------------------------------------------ SHAP/LIME
    st.subheader("Feature contributions")
    panes = st.columns(2)
    with panes[0]:
        shap_payload = payload.get("shap")
        if shap_payload:
            tables.chart_with_table(
                charts.contribution_chart(
                    shap_payload["contributions"], title="SHAP", top_k=int(top_k),
                ),
                tables.contribution_table(shap_payload["contributions"]),
                key="ax_shap_chart",
                meta=f"{shap_payload['explainer']} · base {shap_payload['base_value']:.4f}",
            )
        else:
            st.caption("SHAP produced no explanation for this prediction")
    with panes[1]:
        lime_payload = payload.get("lime")
        if lime_payload:
            tables.chart_with_table(
                charts.contribution_chart(
                    lime_payload["contributions"], title="LIME", top_k=int(top_k),
                ),
                tables.contribution_table(lime_payload["contributions"]),
                key="ax_lime_chart",
                meta=(f"surrogate R\u00b2 {lime_payload['r2_score']:.3f} · "
                      f"local {lime_payload['local_prediction']:.3f}"),
            )
        elif use_lime:
            st.caption("LIME produced no explanation for this prediction")
        else:
            st.caption("LIME is switched off for this view")

    for message in payload.get("errors", []):
        st.warning(message)

    # ------------------------------------------------- probabilities + flow
    lower = st.columns([1, 1])
    with lower[0]:
        probabilities = sorted(result.probabilities.items(), key=lambda kv: -kv[1])[:8]
        tables.list_card(
            "Class probabilities",
            ui.ranked_list_html(
                [(label, 100.0 * float(prob)) for label, prob in probabilities],
                fmt="{:.1f}%", label_html=ui.attack_badge,
            ),
            key="ax_proba", meta="top 8",
        )
    with lower[1]:
        with tables.card("ax_features"):
            ui.render(ui.card_header_html("Flow features used by the model"))
            shown = payload.get("top_features") or []
            names = [str(entry.get("feature")) for entry in shown] or list(result.features)[:12]
            rows = [
                {"Feature": label_for(name), "Canonical name": name,
                 "Value": result.features.get(name)}
                for name in names
            ]
            tables.dataframe(pd.DataFrame(rows), height=320)

    # --------------------------------------------- persist the explanation
    if ui.secondary_button("Save this explanation to the alert history"):
        try:
            database.update_prediction_explanation(
                int(selected),
                narrative.get("text", ""),
                payload.get("top_features", []),
                payload.get("explainer", "SHAP"),
            )
            st.success(f"Explanation stored with prediction #{int(selected)}")
        except NIDSError as exc:
            st.error(str(exc))


__all__ = ["render", "PAGE_TITLE"]
