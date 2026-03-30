"""Inference pipeline using the Snowflake Model Registry."""
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.ml.registry import Registry

from datetime import datetime
import pandas as pd
from utils import (
    get_inference_config, get_feature_config,
    create_session, get_fully_qualified_name
)

infer_cfg = get_inference_config()
feature_cfg = get_feature_config()

GRAIN = feature_cfg["partition_col"]
TARGET = feature_cfg["target_col"]
TIME = feature_cfg["time_col"]
EXCLUDE_COLS = [GRAIN, TARGET, TIME]


def get_registry_model(session: Session, model_name: str = "MMT_POC"):
    """Get the default version of the model from the registry."""
    reg = Registry(
        session=session,
        database_name=session.get_current_database(),
        schema_name=session.get_current_schema(),
    )
    model = reg.get_model(model_name)
    mv = model.default
    print(f"   Model: {model_name}")
    print(f"   Version: {mv.version_name}")
    return mv


def prepare_data(session: Session):
    """Load test data for inference."""
    infer_data = session.table("TEST_DATA")
    keep_cols = [c for c in infer_data.columns if c != TARGET]
    infer_input = infer_data.select(keep_cols)
    print(f"   Rows to score: {infer_data.count()}")
    return infer_data, infer_input


def execute_inference(session: Session, mv, infer_data, infer_input):
    """Run inference using the registered model version."""
    print(f"\n🚀 Running inference via Model Registry")

    raw_pred = mv.run(infer_input, function_name="predict")

    pred_df = raw_pred.select(
        F.col(f"OUTPUT_{GRAIN}").alias(GRAIN),
        F.col(f"OUTPUT_{TIME}").alias(TIME),
        F.col(f"PRED_{TARGET}").alias("PREDICTION"),
    )

    actuals = infer_data.select(GRAIN, TIME, TARGET)
    result = pred_df.join(actuals, on=[GRAIN, TIME])

    result.write.mode("overwrite").save_as_table("PREDICTIONS")
    print(f"   Predictions written to PREDICTIONS table")
    return result


def show_sample_predictions(session: Session):
    """Print sample rows from PREDICTIONS table."""
    print("\n📈 Sample Predictions:")
    predictions_fqn = get_fully_qualified_name("PREDICTIONS")
    session.table(predictions_fqn).show(5)


def run_inference(session: Session = None, model_name: str = "MMT_POC") -> str:
    """Entry point: load registry model, run inference, save predictions. Returns inference_run_id."""

    inference_run_id = f"inference_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    if session is None:
        session = create_session(inference_run_id)
        print(f"Connected: {session.get_current_account()}")
    else:
        session.sql(f"ALTER SESSION SET QUERY_TAG = '{inference_run_id}'").collect()

    print(f"📋 Loading model from registry")
    mv = get_registry_model(session, model_name)

    infer_data, infer_input = prepare_data(session)
    execute_inference(session, mv, infer_data, infer_input)
    show_sample_predictions(session)

    return inference_run_id


if __name__ == "__main__":
    run_id = run_inference()
    print(f"RUN_ID={run_id}")
