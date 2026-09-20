# src/ml/serving/onnx.py
"""
ONNX Model Export, Parity Validation & Latency Benchmarking
Converts trained tree models to optimized ONNX binaries for sub-millisecond FastAPI serving.
"""
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from core.logging import get_logger

logger = get_logger(__name__)


class ONNXModelExporter:
    """
    Exports LightGBM models to ONNX format, validates numerical parity against
    native Python inference, and benchmarks sub-millisecond latency profiles.
    """

    @staticmethod
    def export(
        lgb_model,
        feature_names: List[str],
        output_path: str,
        target_opset: int = 15,
    ) -> str:
        """
        Convert LightGBM model to ONNX representation and save to disk.
        """
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        initial_types = [("input", FloatTensorType([None, len(feature_names)]))]

        logger.info(
            f"Converting LightGBM model to ONNX (features={len(feature_names)}, target_opset={target_opset})..."
        )
        start_time = time.perf_counter()

        onnx_model = onnxmltools.convert_lightgbm(
            lgb_model,
            initial_types=initial_types,
            target_opset=target_opset,
        )

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        duration = time.perf_counter() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            f"Successfully exported ONNX model to '{output_path}' ({file_size_mb:.2f} MB in {duration:.3f}s)."
        )
        return output_path

    @staticmethod
    def validate_numerical_parity(
        native_model,
        onnx_path: str,
        sample_input: pd.DataFrame,
        atol: float = 1e-4,
    ) -> Tuple[bool, float]:
        """
        Validate that ONNX Runtime predictions match native Python predictions
        within absolute tolerance threshold.
        """
        import onnxruntime as rt

        logger.info(f"Validating numerical parity on {len(sample_input)} sample rows...")
        session = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Prepare float32 numpy array for ONNX input
        if isinstance(sample_input, pd.DataFrame):
            numeric_df = sample_input.copy()
            for col in numeric_df.select_dtypes(include=["category"]).columns:
                numeric_df[col] = numeric_df[col].cat.codes.astype(np.float32)
            input_array = numeric_df.values.astype(np.float32)
        else:
            input_array = np.array(sample_input, dtype=np.float32)

        native_preds = native_model.predict(sample_input)
        onnx_preds = session.run(None, {input_name: input_array})[0].flatten()

        max_delta = float(np.max(np.abs(native_preds - onnx_preds)))
        is_valid = bool(max_delta <= atol)

        if is_valid:
            logger.info(f"✅ ONNX parity check PASSED! Max delta: {max_delta:.6e} (tolerance <= {atol}).")
        else:
            logger.warning(f"⚠️ ONNX parity delta exceeded tolerance: {max_delta:.6e} > {atol}.")

        return is_valid, max_delta

    @staticmethod
    def benchmark_latency(
        onnx_path: str,
        sample_input: pd.DataFrame,
        num_runs: int = 500,
    ) -> Dict[str, float]:
        """
        Benchmark single-request inference latency across multiple iterations.
        """
        import onnxruntime as rt

        session = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        single_sample = sample_input.iloc[[0]].copy()
        for col in single_sample.select_dtypes(include=["category"]).columns:
            single_sample[col] = single_sample[col].cat.codes.astype(np.float32)
        payload = single_sample.values.astype(np.float32)

        # Warmup session
        for _ in range(20):
            session.run(None, {input_name: payload})

        latencies_ms = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: payload})
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        results = {
            "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
            "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
            "mean_ms": round(float(np.mean(latencies_ms)), 3),
            "throughput_req_per_sec": round(1000.0 / float(np.mean(latencies_ms)), 1),
        }

        logger.info(
            f"ONNX Latency Benchmark ({num_runs} runs) -> P50: {results['p50_ms']}ms | "
            f"P95: {results['p95_ms']}ms | P99: {results['p99_ms']}ms | "
            f"Throughput: {results['throughput_req_per_sec']} req/sec"
        )
        return results
