"""The preprocessing pipeline: leak-free fitting, ordering and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.contract import LIVE_FEATURES, resolve_features
from src.preprocessing.encoder import LabelCodec, PortBucketTransformer
from src.preprocessing.feature_selector import FeatureSelector
from src.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from src.preprocessing.scaler import QuantileClipper, build_scaler
from src.utils.errors import FeatureContractError, PreprocessingError


def _frame(rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {name: rng.random(rows) * 1000 for name in LIVE_FEATURES}
    data["protocol"] = rng.choice([6, 17], rows)
    frame = pd.DataFrame(data)
    frame.loc[0, "flow_bytes_s"] = np.inf
    frame.loc[1, "flow_packets_s"] = np.nan
    frame["constant_probe"] = 1.0
    return frame


def _labels(rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(["BENIGN", "DDoS"], rows))


def test_scaler_factory():
    assert build_scaler("none") is None
    assert build_scaler("standard") is not None
    with pytest.raises(PreprocessingError):
        build_scaler("nonsense")


def test_quantile_clipper_removes_infinities_and_caps_outliers():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.inf, 1e9]})
    clipper = QuantileClipper(0.9).fit(frame)
    values = clipper.transform(frame)
    assert np.isnan(values[3, 0])          # inf became NaN
    assert values[4, 0] < 1e9              # outlier winsorised


def test_quantile_clipper_can_be_disabled():
    frame = pd.DataFrame({"a": [1.0, 1e9]})
    values = QuantileClipper(None).fit(frame).transform(frame)
    assert values[1, 0] == 1e9


def test_port_bucket_transformer():
    out = PortBucketTransformer().fit_transform(pd.DataFrame({"p": [22, 8080, 60000]}))
    assert list(out.reshape(-1)) == ["well_known", "registered", "dynamic"]


def test_label_codec_roundtrip():
    codec = LabelCodec().fit(["BENIGN", "DDoS", "BENIGN", "PortScan"])
    assert codec.classes_ == ["BENIGN", "DDoS", "PortScan"]
    codes = codec.transform(["DDoS", "BENIGN"])
    assert codec.inverse_transform(codes) == ["DDoS", "BENIGN"]
    assert codec.label_of(0) == "BENIGN"
    assert codec.code_of("PortScan") == 2
    assert codec.code_of("Nope") == -1
    with pytest.raises(PreprocessingError):
        codec.transform(["Unseen"])


def test_label_codec_persistence(tmp_path):
    codec = LabelCodec().fit(["BENIGN", "Bot"])
    path = codec.save(tmp_path / "codec.joblib")
    reloaded = LabelCodec.load(path)
    assert reloaded.classes_ == codec.classes_


def test_feature_selector_drops_constant_and_correlated_columns():
    rng = np.random.default_rng(1)
    base = rng.random(200)
    frame = pd.DataFrame(
        {"a": base, "a_copy": base * 2.0 + 1.0, "constant": 5.0, "b": rng.random(200)}
    )
    selector = FeatureSelector(correlation_threshold=0.98).fit(frame)
    assert "constant" not in selector.selected_
    assert "constant" in selector.report.dropped
    assert ("a" in selector.selected_) ^ ("a_copy" in selector.selected_)
    assert "b" in selector.selected_


def test_feature_selector_can_be_disabled():
    frame = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    selector = FeatureSelector(enabled=False).fit(frame)
    assert selector.selected_ == ["a", "b"]


def test_pipeline_fit_transform_produces_finite_output():
    frame = _frame()
    resolved = resolve_features(frame.columns, use_protocol=True)
    pipeline = PreprocessingPipeline(resolved, PreprocessingConfig())
    values = pipeline.fit_transform(frame, _labels(len(frame)))
    assert values.shape[0] == len(frame)
    assert np.isfinite(values).all()
    assert len(pipeline.output_features) == values.shape[1]


def test_pipeline_enforces_feature_order_on_transform():
    frame = _frame()
    resolved = resolve_features(frame.columns, use_protocol=True)
    pipeline = PreprocessingPipeline(resolved, PreprocessingConfig()).fit(
        frame, _labels(len(frame))
    )
    shuffled = frame[list(reversed(list(frame.columns)))]
    assert np.allclose(pipeline.transform(frame), pipeline.transform(shuffled))


def test_pipeline_rejects_records_missing_contract_features():
    frame = _frame()
    resolved = resolve_features(frame.columns, use_protocol=True)
    pipeline = PreprocessingPipeline(resolved, PreprocessingConfig()).fit(
        frame, _labels(len(frame))
    )
    with pytest.raises(FeatureContractError):
        pipeline.transform(frame.drop(columns=["flow_duration"]))


def test_pipeline_transform_before_fit_is_an_error():
    frame = _frame(10)
    resolved = resolve_features(frame.columns, use_protocol=True)
    with pytest.raises(PreprocessingError):
        PreprocessingPipeline(resolved, PreprocessingConfig()).transform(frame)


def test_pipeline_persistence_roundtrip(tmp_path):
    frame = _frame()
    resolved = resolve_features(frame.columns, use_protocol=True)
    pipeline = PreprocessingPipeline(resolved, PreprocessingConfig()).fit(
        frame, _labels(len(frame))
    )
    pipeline.save(tmp_path)
    reloaded = PreprocessingPipeline.load(tmp_path)
    assert reloaded.input_features == pipeline.input_features
    assert reloaded.output_features == pipeline.output_features
    assert np.allclose(reloaded.transform(frame), pipeline.transform(frame))
    assert (tmp_path / "feature_contract.json").exists()


def test_pipeline_missing_artefact_raises(tmp_path):
    with pytest.raises(PreprocessingError):
        PreprocessingPipeline.load(tmp_path / "absent")


def test_single_record_transform_matches_batch():
    frame = _frame(50)
    resolved = resolve_features(frame.columns, use_protocol=True)
    pipeline = PreprocessingPipeline(resolved, PreprocessingConfig()).fit(
        frame, _labels(len(frame))
    )
    batch = pipeline.transform(frame)
    single = pipeline.transform([frame.iloc[3].to_dict()])
    assert np.allclose(batch[3], single[0])
