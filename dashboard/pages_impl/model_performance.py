"""MODEL PERFORMANCE - the measured training and evaluation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.components import charts, compat, metrics, tables, ui
from dashboard.state import (
    load_dataset_profile,
    load_explainer,
    load_predictor,
    load_training_metadata,
    load_training_metrics,
)
from dashboard.styles import theme
from src.features.contract import contract_summary

PAGE_TITLE = "Model Performance"


def _candidate_frame(metrics_payload: dict, split: str) -> pd.DataFrame:
    rows = []
    for candidate in metrics_payload.get("candidates", []):
        result = candidate.get(split)
        if not result:
            rows.append(
                {"model": candidate["name"], "status": "failed",
                 "error": candidate.get("error", "")}
            )
            continue
        rows.append(
            {
                "model": candidate["name"],
                "status": "trained",
                "accuracy": round(result["accuracy"], 4),
                "balanced_accuracy": round(result["balanced_accuracy"], 4),
                "precision_macro": round(result["precision_macro"], 4),
                "recall_macro": round(result["recall_macro"], 4),
                "macro_f1": round(result["f1_macro"], 4),
                "weighted_f1": round(result["f1_weighted"], 4),
                "roc_auc_ovr": (
                    round(result["roc_auc_ovr_macro"], 4)
                    if result.get("roc_auc_ovr_macro") is not None else None
                ),
                "detection_rate": round(result["binary"]["detection_rate"], 4),
                "false_positive_rate": round(result["binary"]["false_positive_rate"], 4),
                "fit_seconds": candidate.get("fit_seconds"),
                "latency_ms_per_flow": round(result["predict_latency_ms"], 5),
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    metrics_payload = load_training_metrics()
    metadata = load_training_metadata()

    if not metrics_payload or not metadata:
        st.info(
            "No training results are stored yet. Run:\n\n"
            "```\npython scripts/train_model.py\n```"
        )
        return

    best = metrics_payload.get("best_model", {})
    test = best.get("test") or {}
    validation = best.get("validation") or {}

    source = Path(str(metadata.get("dataset", {}).get("source_directory", "")))
    if source.name == "fixture":
        st.warning(
            "**These numbers come from synthetic quickstart data.** They show that the "
            "pipeline runs, and say nothing about CIC-IDS2017. Don't quote them in a report: "
            "install the real dataset and run `python scripts/train_model.py`."
        )

    metrics.status_strip(
        [
            (f"selected · {best.get('name', '?')}", theme.NODE_GREEN),
            (f"metric · {metrics_payload.get('primary_metric', 'macro_f1')}", None),
            (f"trained {metadata.get('created_at', 'unknown')}", None),
        ]
    )

    metrics.stat_row(
        [
            {"label": "Test accuracy", "value": f"{test.get('accuracy', 0):.4f}",
             "delta": "micro average over all flows"},
            {"label": "Macro F1", "value": f"{test.get('f1_macro', 0):.4f}",
             "delta": "every class weighted equally", "highlight": True},
            {"label": "Detection rate", "value": f"{(test.get('binary') or {}).get('detection_rate', 0):.4f}",
             "delta": "recall over all attack classes"},
            {"label": "False-positive rate", "value": f"{(test.get('binary') or {}).get('false_positive_rate', 0):.4f}",
             "delta": "benign flows wrongly alerted"},
        ]
    )

    metrics.panel(
        "How to read these numbers",
        metrics_payload.get("averaging_note", "")
        + "<br><br><b>Selection rationale:</b> "
        + metadata.get("selection", {}).get("rationale", "")
        + "<br><br>The validation split chose the model; the test split was scored "
          "once, afterwards, and never influenced any decision.",
    )

    tabs = st.tabs(
        ["Model comparison", "Confusion matrix", "Classification report",
         "Per-class metrics", "Dataset", "Feature contract", "Global importance"]
    )

    # ---------------------------------------------------- model comparison
    with tabs[0]:
        st.markdown("###### Validation split (used for model selection)")
        validation_frame = _candidate_frame(metrics_payload, "validation")
        tables.dataframe(validation_frame)
        st.markdown("###### Test split (final, unbiased estimate)")
        test_frame = _candidate_frame(metrics_payload, "test")
        tables.dataframe(test_frame)

        if not test_frame.empty and "macro_f1" in test_frame.columns:
            scored = test_frame.dropna(subset=["macro_f1"])
            tables.list_card(
                "Macro F1 on the test split",
                ui.ranked_list_html(zip(scored["model"], scored["macro_f1"]), fmt="{:.4f}"),
                key="mp_macro", meta="higher is better",
            )

    # ------------------------------------------------------ confusion matrix
    with tabs[1]:
        if test.get("confusion_matrix"):
            with tables.card("mp_cm"):
                ui.render(ui.card_header_html("Confusion matrix", "test split"))
                normalise = st.toggle("Normalise by true class", value=True, key="mp_norm")
                figure = charts.confusion_heatmap(
                    test["confusion_matrix"], test["class_names"], normalise=normalise
                )
                figure.update_layout(title_text="", margin_t=8)
                frame = pd.DataFrame(
                    test["confusion_matrix"],
                    index=test["class_names"],
                    columns=test["class_names"],
                )
                compat.plotly_chart(figure, key="mp_cm")
                with st.expander("Data table"):
                    compat.dataframe(frame, hide_index=False)
        else:
            st.caption("No confusion matrix stored.")

    # -------------------------------------------------- classification report
    with tabs[2]:
        st.code(test.get("classification_report_text", "not available"), language="text")
        st.caption(
            "Rows are true classes. 'support' is the number of test flows of that "
            "class. Macro average is the unweighted mean over classes."
        )

    # -------------------------------------------------------- per-class table
    with tabs[3]:
        per_class = pd.DataFrame(test.get("per_class", []))
        if per_class.empty:
            st.caption("No per-class metrics stored.")
        else:
            tables.dataframe(per_class)
            tables.list_card(
                "Recall per class",
                ui.ranked_list_html(
                    zip(per_class["class"], per_class["recall"]), fmt="{:.3f}",
                    label_html=ui.attack_badge, accent_first=False,
                ),
                key="mp_recall", meta="test split",
            )

    # ---------------------------------------------------------------- dataset
    with tabs[4]:
        dataset = metadata.get("dataset", {})
        metrics.stat_row(
            [
                {"label": "Rows available", "value": f"{dataset.get('rows_available', 0):,}"},
                {"label": "Rows used", "value": f"{dataset.get('rows_loaded', 0):,}"},
                {"label": "Classes", "value": f"{metadata.get('n_classes', 0)}"},
                {"label": "Model inputs", "value": f"{len(metadata.get('input_features', []))}"},
            ]
        )
        st.caption("Source directory: " + str(dataset.get("source_directory", "")))
        st.caption("Files used: " + (", ".join(dataset.get("files", [])) or "none recorded"))

        counts = dataset.get("class_counts_loaded", {})
        if counts:
            frame = pd.DataFrame(
                [{"class": k, "rows": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
            )
            tables.list_card(
                "Training-set class distribution",
                ui.ranked_list_html(zip(frame["class"], frame["rows"]), label_html=ui.attack_badge),
                key="mp_classes", meta="rows",
            )

        cleaning = dataset.get("cleaning", {})
        if cleaning:
            st.markdown("###### Cleaning report")
            tables.dataframe(
                pd.DataFrame(
                    [
                        {"item": "rows in", "value": cleaning.get("rows_in")},
                        {"item": "rows out", "value": cleaning.get("rows_out")},
                        {"item": "duplicates removed", "value": cleaning.get("duplicates_removed")},
                        {"item": "infinite values replaced",
                         "value": cleaning.get("infinite_values_replaced")},
                        {"item": "rows with null label", "value": cleaning.get("rows_with_null_label")},
                        {"item": "missing values remaining",
                         "value": cleaning.get("missing_values_remaining")},
                        {"item": "columns dropped",
                         "value": len(cleaning.get("columns_dropped", {}))},
                    ]
                )
            )

        profile = load_dataset_profile()
        if profile:
            with st.expander("Full dataset profile (reports/dataset_profile.json)"):
                st.json(
                    {
                        k: v for k, v in profile.items()
                        if k not in ("feature_statistics",)
                    }
                )
                stats_frame = pd.DataFrame(profile.get("feature_statistics", []))
                if not stats_frame.empty:
                    compat.dataframe(stats_frame)

    # -------------------------------------------------------- feature contract
    with tabs[5]:
        st.markdown(
            "The contract below is the single source of truth shared by training "
            "and live capture. Features marked **excluded** are deliberately not "
            "modelled - the reason is given for each."
        )
        summary = contract_summary()
        selection = st.radio(
            "Show", ["modelled", "excluded", "all"], horizontal=True, key="mp_contract"
        )
        if selection != "all":
            summary = summary[summary["status"] == selection]
        tables.dataframe(summary, height=520)

        policy = metadata.get("feature_policy", {})
        st.caption(
            f"Destination port used as an input: **{policy.get('use_destination_port')}** · "
            f"Protocol used: **{policy.get('use_protocol')}** · "
            f"Packet-length basis: **{policy.get('packet_length_basis')}**"
        )
        resolved = policy.get("resolved", {})
        if resolved.get("missing"):
            st.warning(
                "These contract features were absent from the training CSVs and "
                "were skipped: " + ", ".join(resolved["missing"])
            )

    # ------------------------------------------------------ global importance
    with tabs[6]:
        st.markdown(
            "Mean absolute SHAP value over the stored background sample - the "
            "model's *global* feature ranking, as opposed to the per-flow "
            "attributions on the AI Explanation page."
        )
        if st.button("Compute global SHAP importance", key="mp_global"):
            service = load_explainer()
            predictor = load_predictor()
            if service is None or predictor is None or not service.shap_available:
                st.error("SHAP is not available for the selected model.")
            else:
                with st.spinner("Computing SHAP values over the background sample ..."):
                    try:
                        ranking = service.shap.global_importance(top_k=15)
                    except Exception as exc:
                        st.error(f"Global importance failed: {exc}")
                        ranking = []
                if ranking:
                    st.session_state["mp_ranking"] = ranking
        ranking = st.session_state.get("mp_ranking", [])
        if ranking:
            tables.list_card(
                "Top contributing features",
                ui.ranked_list_html(
                    [(r.get("label") or r.get("feature", ""), r.get("importance", 0.0))
                     for r in ranking[:15]],
                    fmt="{:.4f}",
                ),
                key="mp_global_chart", meta="mean |SHAP|",
            )


__all__ = ["render", "PAGE_TITLE"]
