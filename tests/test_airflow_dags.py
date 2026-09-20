# tests/test_airflow_dags.py
"""
Tests for Airflow DAG definitions and integrity:
- Compiles each DAG file to verify Python syntax and AST validity
- Validates DAG IDs, task definitions, and operators
- Ensures no legacy shim imports or deprecated modules are referenced
"""
import ast
import os
import glob
import pytest


DAG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "airflow", "dags")


def get_dag_files():
    """Retrieve all Python DAG files from airflow/dags."""
    return glob.glob(os.path.join(DAG_DIR, "*.py"))


def test_dag_files_exist():
    """Ensure that the expected DAG files are present."""
    files = [os.path.basename(f) for f in get_dag_files()]
    assert "elt_pipeline_dag.py" in files
    assert "ml_drift_monitoring_dag.py" in files
    assert "ml_retrain_dag.py" in files


@pytest.mark.parametrize("dag_path", get_dag_files())
def test_dag_python_syntax(dag_path):
    """Verify that every DAG file is syntactically valid Python code."""
    with open(dag_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Must compile without raising SyntaxError
    compiled = compile(source, dag_path, "exec")
    assert compiled is not None

    # Must parse cleanly into AST
    tree = ast.parse(source, filename=dag_path)
    assert tree is not None


def test_no_legacy_shim_imports_in_dags():
    """Ensure no DAG imports from removed legacy shims (ml.data_loader, ml.drift, etc.)."""
    forbidden_substrings = [
        "from ml.data_loader",
        "import ml.data_loader",
        "from ml.drift",
        "import ml.drift",
        "from demand_prediction",
        "import demand_prediction",
        "from network_location_demand_analysis",
        "import network_location_demand_analysis",
        "Deprecated shim",
    ]

    for dag_path in get_dag_files():
        with open(dag_path, "r", encoding="utf-8") as f:
            content = f.read()
        for forbidden in forbidden_substrings:
            assert forbidden not in content, (
                f"Found forbidden legacy reference '{forbidden}' in {dag_path}"
            )


def test_drift_monitoring_dag_structure():
    """Verify ml_drift_monitoring_dag contains gated short-circuit and retrain trigger."""
    dag_file = os.path.join(DAG_DIR, "ml_drift_monitoring_dag.py")
    with open(dag_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'dag_id="ml_drift_monitoring_dag"' in content
    assert "ShortCircuitOperator" in content
    assert "TriggerDagRunOperator" in content
    assert 'trigger_dag_id="ml_model_retrain_and_evaluate_dag"' in content
    assert "DriftMonitoringPipeline" in content


def test_retrain_dag_structure():
    """Verify ml_retrain_dag uses direct MLTrainingPipeline and SpatialNetworkAnalyzer."""
    dag_file = os.path.join(DAG_DIR, "ml_retrain_dag.py")
    with open(dag_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'dag_id="ml_model_retrain_and_evaluate_dag"' in content
    assert "MLTrainingPipeline" in content
    assert "SpatialNetworkAnalyzer" in content
    assert "schedule_interval=None" in content
