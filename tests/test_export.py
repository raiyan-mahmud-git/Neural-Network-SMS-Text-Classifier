"""Ensure export failures are reported without fabricated benchmark values."""

from types import SimpleNamespace

import export


class DummyModel:
    def predict(self, inputs, verbose=0):
        return [[0.25]]


def test_failed_tflite_conversion_has_no_estimated_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "get_inference_model", lambda: DummyModel())
    monkeypatch.setattr(export.config, "BEST_TFLITE_PATH", tmp_path / "best_model.tflite")
    monkeypatch.setattr(
        export.tf.lite.TFLiteConverter,
        "from_keras_model",
        lambda model: SimpleNamespace(
            target_spec=SimpleNamespace(supported_ops=[]),
            convert=lambda: (_ for _ in ()).throw(RuntimeError("conversion unsupported")),
        ),
    )

    result = export.export_models_and_benchmark(num_benchmark_runs=1)

    assert result["keras_model"]["file_size_kb"] > 0
    tflite = result["tflite_model"]
    assert tflite["status"] == "export_failed"
    assert tflite["file_size_kb"] is None
    assert tflite["latency_ms_per_sample"] is None
    assert tflite["throughput_samples_per_sec"] is None
    assert not (tmp_path / "best_model.tflite").exists()
