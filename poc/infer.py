"""Distributed inference pipeline using DPF to load models and generate predictions."""
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F

from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.modeling.distributors.distributed_partition_function.dpf import DPF
from snowflake.ml.modeling.distributors.many_model import PickleSerde
from snowflake.ml.modeling.distributors.distributed_partition_function.entities import ExecutionOptions

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from utils import (
    get_inference_config, get_feature_config,
    create_session, get_fully_qualified_name, get_stage_path,
    stage_data_partitioned, copy_from_stage_to_table
)

infer_cfg = get_inference_config()
feature_cfg = get_feature_config()

serde = PickleSerde()

GRAIN = feature_cfg["partition_col"]
TARGET = feature_cfg["target_col"]
TIME = feature_cfg["time_col"]
EXCLUDE_COLS = [GRAIN, TARGET, TIME]
TEST_PCT = feature_cfg["test_pct"]

def predict_partition(data_connector: DataConnector, context):
    """DPF worker: load model from stage, generate predictions, upload results.
    Update this function for specific use cases"""
    import pandas as pd
    import numpy as np

    df = data_connector.to_pandas()
    if df.empty:
        return None
        
    model_path = df["STAGE_PATH"].unique()[0]
    print(model_path)
    model = context.download_from_stage(
        serde.filename,
        stage_path=model_path,
        read_function=serde.read,
    )

    df = df.sort_values(TIME)
        
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        
    X = df[model.feature_names_in_].astype("float32")

    preds = model.predict(X)
        
    out = df[[GRAIN, TIME, TARGET]].copy()
    out["PREDICTION"] = preds
    
    context.upload_to_stage(out, "predictions.parquet",
        write_function=lambda pdf, path: pdf.to_parquet(path, index=False))

def get_latest_model_version(session: Session) -> str:
    """Return the most recent active model version from MODEL_CATALOG."""
    result = session.sql(f"""
        SELECT MODEL_VERSION FROM MODEL_CATALOG
        WHERE IS_ACTIVE = TRUE ORDER BY TRAINED_AT DESC LIMIT 1
    """).collect()
    return result[0]['MODEL_VERSION']


def prepare_data(session: Session):
    """Join test data with model paths, stage for distributed inference."""    
    infer_data = session.table("TEST_DATA")
    catalog = session.table("MODEL_CATALOG").filter(F.col("IS_ACTIVE")).select(GRAIN, "STAGE_PATH")
    
    infer_data = infer_data.join(catalog, on = GRAIN)
    infer_data.write.mode("overwrite").save_as_table("INFER_TRANSIENT", table_type="temporary")
    
    stage_data_partitioned(session, "INFER_TRANSIENT", "infer_data", partition_col=GRAIN)
    
    print(f"   Active models: {catalog.count()}")
    print(f"   Rows to score: {infer_data.count()}")
    return infer_data

    
def execute_inference(session: Session, train_run_id: str, inference_run_id: str):
    """Run DPF inference job across all partitions."""
    
    print(f"\n🚀 Starting Many Model Inference")
    print(f"   Using models from: {train_run_id}")
    
    inf_dpf = DPF(predict_partition, get_stage_path())
    inf_run = inf_dpf.run_from_stage(
        stage_location=get_stage_path("infer_data"),
        run_id=inference_run_id,
        on_existing_artifacts="overwrite",
    )
        
    inf_status = inf_run.wait()
    print(f"\n   Inference status: {inf_status}")
    return inf_status


def collect_predictions(session: Session, inference_run_id: str):
    """Copy prediction parquet files from stage into PREDICTIONS table."""
    copy_from_stage_to_table(session, "PREDICTIONS", f"{inference_run_id}/infer_data")


def show_sample_predictions(session: Session):
    """Print sample rows from PREDICTIONS table."""
    print("\n📈 Sample Predictions:")
    predictions_fqn = get_fully_qualified_name("PREDICTIONS")
    session.table(predictions_fqn).show(5)


def run_inference(session: Session = None):
    """Entry point: prepare data, run inference, collect predictions."""

    if session is None:
        session = create_session("INFER")
        print(f"Connected: {session.get_current_account()}")
    
    latest_version = get_latest_model_version(session)
    train_run_id = f"training_{latest_version}"
    inference_run_id = f"inference_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
    
    print(f"📋 Using models from: {train_run_id}")
    print(f"   Inference run ID:  {inference_run_id}")
    
    prepare_data(session)
    execute_inference(session, train_run_id, inference_run_id)
    collect_predictions(session, inference_run_id)
    show_sample_predictions(session)


if __name__ == "__main__":
    run_inference()
