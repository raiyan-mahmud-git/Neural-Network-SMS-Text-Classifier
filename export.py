"""
Model Export & Deployment Optimization: .keras vs TensorFlow Lite (TFLite).

Measures and compares:
1. Disk footprint (File size in KB/MB)
2. Single-sample inference latency (milliseconds)
3. Throughput (inferences per second)

"""

import logging
import time
from typing import Any

import numpy as np
import tensorflow as tf

import config
import data
from predict import get_inference_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_models_and_benchmark(num_benchmark_runs: int = 100) -> dict[str, Any]:
    """Export the verified Keras artifact and report only measured results."""
    if num_benchmark_runs < 1:
        raise ValueError("num_benchmark_runs must be at least 1")

    model = get_inference_model()
    keras_path = config.BEST_MODEL_PATH
    keras_size_kb = keras_path.stat().st_size / 1024.0

    sample_text = data.clean_text(
        "Congratulations! You have won £1000 cash. Call 08712460324 to claim."
    )
    sample_input = np.array([sample_text], dtype=object)

    for _ in range(10):
        model.predict(sample_input, verbose=0)
    started_at = time.perf_counter()
    for _ in range(num_benchmark_runs):
        model.predict(sample_input, verbose=0)
    keras_latency_ms = (time.perf_counter() - started_at) / num_benchmark_runs * 1000.0

    keras_result = {
        "format": ".keras (Native Keras 3 Archive)",
        "file_size_kb": round(keras_size_kb, 2),
        "latency_ms_per_sample": round(keras_latency_ms, 2),
        "throughput_samples_per_sec": round(1000.0 / keras_latency_ms, 1),
    }
    tflite_result: dict[str, Any] = {
        "format": ".tflite (FlatBuffer Edge Format)",
        "status": "not_exported",
        "file_size_kb": None,
        "latency_ms_per_sample": None,
        "throughput_samples_per_sec": None,
    }

    tflite_path = config.BEST_TFLITE_PATH
    tflite_path.unlink(missing_ok=True)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converted_model = converter.convert()
        temporary_path = tflite_path.with_suffix(".tflite.tmp")
        temporary_path.write_bytes(converted_model)
        temporary_path.replace(tflite_path)
        tflite_result["file_size_kb"] = round(tflite_path.stat().st_size / 1024.0, 2)
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        string_input = np.array([sample_text.encode("utf-8")], dtype=object)

        for _ in range(10):
            interpreter.set_tensor(input_details[0]["index"], string_input)
            interpreter.invoke()
        started_at = time.perf_counter()
        for _ in range(num_benchmark_runs):
            interpreter.set_tensor(input_details[0]["index"], string_input)
            interpreter.invoke()
            interpreter.get_tensor(output_details[0]["index"])
        tflite_latency_ms = (time.perf_counter() - started_at) / num_benchmark_runs * 1000.0
        tflite_result.update({
            "status": "exported_and_benchmarked",
            "latency_ms_per_sample": round(tflite_latency_ms, 2),
            "throughput_samples_per_sec": round(1000.0 / tflite_latency_ms, 1),
        })
    except Exception as exc:
        temporary_path = tflite_path.with_suffix(".tflite.tmp")
        temporary_path.unlink(missing_ok=True)
        if not tflite_path.exists():
            tflite_result["status"] = "export_failed"
        else:
            tflite_result["status"] = "benchmark_failed"
        tflite_result["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("TFLite result: %s", tflite_result["status"], exc_info=True)

    results = {"keras_model": keras_result, "tflite_model": tflite_result}
    logger.info("=== Deployment Optimization Benchmark ===")
    logger.info("%s", results)
    return results

if __name__ == "__main__":
    export_models_and_benchmark()
