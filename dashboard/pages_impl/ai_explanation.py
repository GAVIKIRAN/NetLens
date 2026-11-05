"""AI EXPLANATION - SHAP and LIME for one selected prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import charts, metrics, tables
from dashboard.state import load_database, load_explainer, load_predictor, load_settings
from src.features.contract import label_for
from src.utils.errors import ExplainerError, NIDSError
from src.xai.local_llm import DEFAULT_BASE_URL, OllamaClient, review_prediction

PAGE_TITLE = "AI Explanation"


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
    try:
        client = OllamaClient(
            str(config.get("base_url", DEFAULT_BASE_URL)),
            timeout=float(config.get("timeout_seconds", 180)),
        )
        installed = client.list_models()
    except ExplainerError as exc:
        st.info(f"{exc} The rest of this page works without it.")
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
    if st.button("Review this prediction", key="ax_llm_run"):
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
    st.markdown(
        " ".join(
            [
                metrics.pill(
                    f"SHAP · {status['shap_explainer']}",
                    "ok" if status["shap_available"] else "danger",
                ),
                metrics.pill(
                    "LIME ready" if status["lime_available"] else "LIME unavailable",
                    "ok" if status["lime_available"] else "danger",
                ),
                metrics.pill(
                    "inline explanations on" if status["inline_explanations"]
                    else "on-demand explanations only",
                    "info",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )
    for message in status.get("errors", []):
        st.warning(message)

    # ----------------------------------------------------- pick a prediction
    st.markdown("##### Select a prediction")
    columns = st.columns([1, 1, 1, 2])
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
            "No stored predictions match. Start monitoring from the **Live Monitor** "
            "page (DEMO mode is safe) and come back."
        )
        return

    def _describe(prediction_id: int) -> str:
        row = candidates[candidates["id"] == prediction_id].iloc[0]
        return (
            f"#{prediction_id} · {row['prediction']} · "
            f"conf {float(row['confidence']):.0%} · risk {float(row['risk_score']):.0f} "
            f"· {row['src_ip']}→{row['dst_ip']}"
        )

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

    with st.spinner("Computing explanations ..."):
        try:
            result = _rebuild_result(record, predictor)
            payload = service.explain_prediction(
                result, use_shap=True, use_lime=bool(use_lime), top_k=int(top_k)
            )
        except NIDSError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - library specific
            st.error(f"Explanation failed: {exc}")
            return

    # ------------------------------------------------------------- headline
    st.divider()
    header = st.columns([1, 1, 2])
    with header[0]:
        metrics.stat_tile(
            "Prediction", result.predicted_label,
            f"stored as '{record.get('prediction')}'",
            accent="red" if result.is_attack else "low",
        )
    with header[1]:
        metrics.stat_tile(
            "Risk band", str(result.risk.level).title(),
            f"score {result.risk.score:.0f}/100", accent=str(result.risk.level).lower(),
        )
    with header[2]:
        metrics.confidence_vs_risk(
            result.confidence, result.risk.score, result.risk.level
        )

    if record.get("final_classification"):
        st.markdown(
            f"**IDS decision: {record['final_classification']}**  \n"
            f"Model: {record.get('prediction')} · "
            f"Behaviour: {record.get('behaviour_result') or 'None'}  \n"
            f"{record.get('detection_reason') or ''}"
        )

    if record.get("ground_truth"):
        st.caption(
            f"Demo ground-truth label for this record: **{record['ground_truth']}** "
            "(shown for transparency - the model was not told this)."
        )
    if result.threshold_applied:
        st.warning(
            f"The model's highest-scoring class was **{result.argmax_label}** at "
            f"{result.argmax_confidence:.1%}, below the configured decision "
            f"threshold of {predictor.attack_threshold:.0%}, so this flow is "
            "reported as benign."
        )

    st.markdown("###### Risk formula for this flow")
    st.code(result.risk.formula, language="text")

    # ------------------------------------------------------------ narrative
    st.subheader("Human-readable explanation")
    narrative = payload.get("narrative") or {}
    with st.container(key="card_narrative"):
        st.markdown(f"**{narrative.get('headline', '')}**")
        st.markdown(narrative.get("summary", ""))
        for line in narrative.get("bullet_points", []):
            st.markdown(f"- {line}")
        st.caption(narrative.get("caveat", ""))

    _render_local_review(settings, record, payload, int(selected))

    # ------------------------------------------------------------ SHAP/LIME
    st.divider()
    panes = st.columns(2)
    with panes[0]:
        st.subheader("SHAP")
        shap_payload = payload.get("shap")
        if shap_payload:
            st.caption(
                f"Explainer: **{shap_payload['explainer']}** · base value "
                f"{shap_payload['base_value']:.4f}"
            )
            tables.chart_with_table(
                charts.contribution_chart(
                    shap_payload["contributions"],
                    title="SHAP contributions", top_k=int(top_k),
                ),
                tables.contribution_table(shap_payload["contributions"]),
                key="ax_shap_chart",
            )
        else:
            st.caption("SHAP produced no explanation for this prediction.")
    with panes[1]:
        st.subheader("LIME")
        lime_payload = payload.get("lime")
        if lime_payload:
            st.caption(
                f"Local surrogate fit R² = {lime_payload['r2_score']:.3f} · "
                f"local prediction {lime_payload['local_prediction']:.3f}"
            )
            tables.chart_with_table(
                charts.contribution_chart(
                    lime_payload["contributions"],
                    title="LIME contributions", top_k=int(top_k),
                ),
                tables.contribution_table(lime_payload["contributions"]),
                key="ax_lime_chart",
            )
        elif use_lime:
            st.caption("LIME produced no explanation for this prediction.")
        else:
            st.caption("LIME is switched off for this view.")

    for message in payload.get("errors", []):
        st.warning(message)

    # ------------------------------------------------- probabilities + flow
    st.divider()
    lower = st.columns([1, 1])
    with lower[0]:
        probabilities = result.probabilities
        tables.chart_with_table(
            charts.probability_bar(probabilities),
            pd.DataFrame(
                [{"class": k, "probability": round(v, 6)} for k, v in
                 sorted(probabilities.items(), key=lambda kv: -kv[1])]
            ),
            key="ax_proba",
        )
    with lower[1]:
        st.markdown("###### Flow features used by the model")
        shown = payload.get("top_features") or []
        names = [str(entry.get("feature")) for entry in shown] or list(result.features)[:12]
        rows = [
            {
                "Feature": label_for(name),
                "Canonical name": name,
                "Value": result.features.get(name),
            }
            for name in names
        ]
        tables.dataframe(pd.DataFrame(rows), height=360)

    # --------------------------------------------- persist the explanation
    if st.button("Save this explanation to the alert history"):
        try:
            database.update_prediction_explanation(
                int(selected),
                narrative.get("text", ""),
                payload.get("top_features", []),
                payload.get("explainer", "SHAP"),
            )
            st.success("Explanation stored with prediction #%d." % int(selected))
        except NIDSError as exc:
            st.error(str(exc))


__all__ = ["render", "PAGE_TITLE"]
