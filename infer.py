"""Inference and drift detection pipeline for IC volume forecasting."""
from snowflake.snowpark import Session

from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.modeling.distributors.distributed_partition_function.dpf import DPF
from snowflake.ml.modeling.distributors.many_model import PickleSerde
from snowflake.ml.modeling.distributors.distributed_partition_function.entities import ExecutionOptions
from datetime import date, datetime, timedelta
import modin.pandas as pd
import snowflake.snowpark.modin.plugin
import numpy as np
from utils import (
    get_inference_config, get_tables_config,
    create_session, get_table, get_fully_qualified_name, get_stage_path,
    stage_data_partitioned, copy_from_stage_to_table
)

infer_cfg = get_inference_config()
tables_cfg = get_tables_config()

serde = PickleSerde()

GRAIN = ["MANAGED_BY", "SALES_OFFICE", "STORE_NUM", "MATERIAL_ID"]
TARGET = "WM_DAILY_CASE_VOLUME"
EXCLUDE_COLS = [
    "ACTUAL_DATE", "STORE_MATERIAL_KEY", "DELIVERY_PLAN",
    "WM_DAILY_VOLUME_EACH", "VOLUME", "ORDERS",
]


def get_latest_model_version(session: Session) -> str:
    """Query MODEL_CATALOG for the most recent active model version."""
    catalog_fqn = get_fully_qualified_name("model_catalog")
    result = session.sql(f"""
        SELECT MODEL_VERSION FROM {catalog_fqn}
        WHERE IS_ACTIVE = TRUE ORDER BY TRAINED_AT DESC LIMIT 1
    """).collect()
    return result[0]['MODEL_VERSION']


def prepare_scoring_data(session: Session, holdout_days: int):
    """Extract recent data from FEATURES_ENRICHED for scoring active partitions."""
    import modin.pandas as pd
    
    features_enriched = get_table("features_enriched")
    catalog = get_table("model_catalog")
    
    active_partitions = catalog[catalog["IS_ACTIVE"]]["STORE_MATERIAL_KEY"].unique()
    features_enriched = features_enriched[features_enriched["STORE_MATERIAL_KEY"].isin(active_partitions)]
    stage_paths = catalog[catalog["IS_ACTIVE"]][["STORE_MATERIAL_KEY","STAGE_PATH"]]
    
    max_date = features_enriched["ACTUAL_DATE"].max()
    cutoff_date = max_date - timedelta(days=holdout_days)
    
    scoring_data = features_enriched[features_enriched["ACTUAL_DATE"] >= cutoff_date]
    scoring_data = scoring_data.merge(stage_paths, on = "STORE_MATERIAL_KEY")
    table_name = get_fully_qualified_name("scoring_transient")
    scoring_data.to_snowflake(table_name, if_exists="replace", index=False, table_type="temp")
    
    stage_data_partitioned(session, "scoring_transient", "scoring_data")
    
    print(f"📅 Scoring window: {cutoff_date} to {max_date}")
    print(f"   Active models: {len(active_partitions):,}")
    print(f"   Rows to score: {len(scoring_data):,}")
    return scoring_data

    
def execute_inference(session: Session, train_run_id: str, inference_run_id: str):
    """Run distributed inference using DPF to load models and generate predictions."""
    def predict_partition(data_connector: DataConnector, context):
        """DPF worker: load model, predict, apply delivery-day grouping."""
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

        df = df.sort_values("ACTUAL_DATE")
        
        exclude = set(GRAIN + EXCLUDE_COLS + [TARGET])
        feature_cols = [c for c in df.columns if c not in exclude]
        
        X = df[model.feature_names_in_].astype("float32")

        log_preds = model.predict(X)
        pred_cases = np.maximum(0, (10 ** log_preds) - 1)
        pred_eaches = pred_cases * 24
        
        out = df[["ACTUAL_DATE", "STORE_NUM", "MATERIAL_ID", "STORE_MATERIAL_KEY",
                    "WM_DAILY_CASE_VOLUME", "DELIVERY_INDICATOR"]].copy()
        out["PRED_LOG_TARGET"] = log_preds
        out["PRED_CASES"] = pred_cases
        out["PRED_EACHES"] = pred_eaches
        
        def group_values(data, grouping_col, sum_col):
            data = data.copy()
            data["_GROUP"] = data[grouping_col][::-1].cumsum()[::-1]
            data[sum_col + "_GROUPED"] = data.groupby("_GROUP")[sum_col].cumsum()
            data[sum_col + "_GROUPED"] = np.where(data[grouping_col] == False, 0, data[sum_col + "_GROUPED"])
            return data.drop(columns=["_GROUP"])
        
        out = group_values(out, "DELIVERY_INDICATOR", "PRED_EACHES")
        out["PRED_VOLUME_GROUPED"] = np.ceil(out["PRED_EACHES_GROUPED"] / 24)
        out["PRED_VOLUME_RAW"] = pred_cases
        
        context.upload_to_stage(out, "predictions.parquet",
                                write_function=lambda pdf, path: pdf.to_parquet(path, index=False))

    predictions_fqn = get_fully_qualified_name("ic_predictions")
    session.sql(f"TRUNCATE TABLE IF EXISTS {predictions_fqn}").collect()
    
    print(f"\n🚀 Starting Many Model Inference")
    print(f"   Using models from: {train_run_id}")
    
    inf_dpf = DPF(predict_partition, get_stage_path())
    inf_run = inf_dpf.run_from_stage(
        stage_location=get_stage_path("scoring_data"),
        run_id=inference_run_id,
        on_existing_artifacts="overwrite",
        execution_options=ExecutionOptions(use_head_node=False, num_cpus_per_worker=1),
    )
        
    inf_status = inf_run.wait()
    print(f"\n   Inference status: {inf_status}")
    return inf_status


def collect_predictions(session: Session, inference_run_id: str):
    """Load prediction results from stage into IC_PREDICTIONS table."""
    table_name = get_fully_qualified_name("ic_predictions")
    copy_from_stage_to_table(session, table_name, f"{inference_run_id}/scoring_data")


def show_sample_predictions(session: Session):
    """Display sample predictions for verification."""
    print("\n📈 Sample Predictions:")
    predictions_fqn = get_fully_qualified_name("ic_predictions")
    session.sql(f"""
        SELECT ACTUAL_DATE, STORE_NUM, MATERIAL_ID, DELIVERY_INDICATOR,
               ROUND(WM_DAILY_CASE_VOLUME, 1) AS ACTUAL_CASES,
               ROUND(PRED_VOLUME_RAW, 1) AS PRED_RAW,
               ROUND(PRED_VOLUME_GROUPED, 1) AS PRED_GROUPED
        FROM {predictions_fqn}
        WHERE DELIVERY_INDICATOR = TRUE
        ORDER BY ACTUAL_DATE DESC LIMIT 15
    """).show()


def run_drift_detection(session: Session, drift_threshold: float, run_id: str):
    """Compute drift metrics by comparing recent MAE to baseline and flag drifted models."""
    print(f"\n📊 Running Drift Detection (threshold: {drift_threshold}x baseline)")
    
    predictions = get_table("ic_predictions")
    catalog = get_table("model_catalog")
    
    today = pd.Timestamp.today().normalize().date()
    cutoff_14d = today - pd.Timedelta(days=14)
    cutoff_7d = today - pd.Timedelta(days=7)
    
    preds_recent = predictions[
        (predictions["ACTUAL_DATE"] >= cutoff_14d) & 
        (predictions["WM_DAILY_CASE_VOLUME"].notna())
    ].copy()
    
    preds_recent["ABS_ERROR"] = (preds_recent["PRED_LOG_TARGET"] - preds_recent["WM_DAILY_CASE_VOLUME"]).abs()
    preds_recent["IS_LAST_7D"] = preds_recent["ACTUAL_DATE"] >= cutoff_7d
    
    print("preds recent", len(preds_recent))

    recent_perf = preds_recent.groupby("STORE_MATERIAL_KEY").agg(
        PREDICTION_COUNT=("STORE_MATERIAL_KEY", "count"),
        ACTUAL_COUNT=("WM_DAILY_CASE_VOLUME", "count"),
        ROLLING_MAE_14D=("ABS_ERROR", "mean"),
    ).reset_index()
    
    preds_7d = preds_recent[preds_recent["IS_LAST_7D"]]
    mae_7d = preds_7d.groupby("STORE_MATERIAL_KEY")["ABS_ERROR"].mean().reset_index()
    mae_7d.columns = ["STORE_MATERIAL_KEY", "ROLLING_MAE_7D"]

    print("recent_perf", len(recent_perf))
    
    recent_perf = recent_perf.merge(mae_7d, on="STORE_MATERIAL_KEY", how="left")
    
    catalog_active = catalog[catalog["IS_ACTIVE"] == True][["STORE_MATERIAL_KEY", "MODEL_VERSION", "TEST_MAE"]]
    
    perf_log = recent_perf.merge(catalog_active, on="STORE_MATERIAL_KEY", how="inner")
    perf_log["RUN_ID"] = run_id
    perf_log["LOG_DATE"] = date.today()
    perf_log["BASELINE_TEST_MAE"] = perf_log["TEST_MAE"]
    perf_log["DRIFT_RATIO"] = perf_log["ROLLING_MAE_14D"] / perf_log["TEST_MAE"].replace(0, np.nan)
    perf_log["IS_DRIFTED"] = perf_log["DRIFT_RATIO"] > drift_threshold
    
    perf_log = perf_log[[
        "RUN_ID", "LOG_DATE", "STORE_MATERIAL_KEY", "MODEL_VERSION", "PREDICTION_COUNT", 
        "ACTUAL_COUNT", "ROLLING_MAE_7D", "ROLLING_MAE_14D", "BASELINE_TEST_MAE", 
        "DRIFT_RATIO", "IS_DRIFTED"
    ]]

    print("perf_log", len(perf_log))
    table_name = get_fully_qualified_name("model_performance_log")
    perf_log.to_snowflake(table_name, if_exists="append", index=False)
    print("✓ MODEL_PERFORMANCE_LOG updated")


def get_drift_summary() -> dict:
    """Query most recent drift detection run results summary."""
    import modin.pandas as pd
    
    perf_log = get_table("model_performance_log")
    if len(perf_log) == 0:
        return {
            'TOTAL_MODELS': 0,
            'DRIFTED_MODELS': 0,
            'AVG_DRIFT_RATIO': None,
            'MAX_DRIFT_RATIO': None,
            'AVG_ROLLING_MAE': None,
        }
    
    latest_date = perf_log["LOG_DATE"].max()
    latest_log = perf_log[perf_log["LOG_DATE"] == latest_date]
    
    return {
        'TOTAL_MODELS': len(latest_log),
        'DRIFTED_MODELS': int(latest_log["IS_DRIFTED"].sum()) if len(latest_log) > 0 else 0,
        'AVG_DRIFT_RATIO': round(latest_log["DRIFT_RATIO"].mean(), 2) if len(latest_log) > 0 else None,
        'MAX_DRIFT_RATIO': round(latest_log["DRIFT_RATIO"].max(), 2) if len(latest_log) > 0 else None,
        'AVG_ROLLING_MAE': round(latest_log["ROLLING_MAE_14D"].mean(), 2) if len(latest_log) > 0 else None,
    }


def print_drift_summary(drift_summary: dict, drift_threshold: float):
    """Print drift detection results and list top drifted models if any."""
    import modin.pandas as pd
    
    total = drift_summary['TOTAL_MODELS'] or 0
    drifted = drift_summary['DRIFTED_MODELS'] or 0
    pct_drifted = (100 * drifted / total) if total > 0 else 0
    
    print(f"\n🎯 Drift Detection Summary:")
    print(f"   Total models:      {total:,}")
    print(f"   Drifted models:    {drifted:,} ({pct_drifted:.1f}%)")
    print(f"   Avg drift ratio:   {drift_summary['AVG_DRIFT_RATIO'] or 0:.2f}x")
    print(f"   Max drift ratio:   {drift_summary['MAX_DRIFT_RATIO'] or 0:.2f}x")
    print(f"   Avg rolling MAE:   {drift_summary['AVG_ROLLING_MAE'] or 0:.2f} cases")
    
    if drifted > 0:
        print(f"\n⚠️ {drifted} models flagged for retraining")
        print("   Run train.py with MODE='SELECTIVE' to retrain\n")
        print("Top 10 most drifted models:")
        
        perf_log = get_table("model_performance_log")
        latest_date = perf_log["LOG_DATE"].max()
        drifted_models = perf_log[(perf_log["LOG_DATE"] == latest_date) & (perf_log["IS_DRIFTED"] == True)]
        top_drifted = drifted_models.nlargest(10, "DRIFT_RATIO")[
            ["STORE_MATERIAL_KEY", "MODEL_VERSION", "ROLLING_MAE_14D", "BASELINE_TEST_MAE", "DRIFT_RATIO"]
        ].copy()
        top_drifted["ROLLING_MAE_14D"] = top_drifted["ROLLING_MAE_14D"].round(2)
        top_drifted["BASELINE_TEST_MAE"] = top_drifted["BASELINE_TEST_MAE"].round(4)
        top_drifted["DRIFT_RATIO"] = top_drifted["DRIFT_RATIO"].round(2)
        print(top_drifted.to_string(index=False))
    else:
        print(f"\n✅ All models within acceptable drift threshold ({drift_threshold}x)")


def print_inference_summary(drift_summary: dict, drift_threshold: float):
    """Print final inference pipeline summary."""
    print("\n" + "=" * 60)
    print("INFERENCE & DRIFT DETECTION COMPLETE")
    print("=" * 60)
    
    predictions = get_table("ic_predictions")
    pred_count = len(predictions)
    
    drifted = drift_summary['DRIFTED_MODELS'] or 0
    total = drift_summary['TOTAL_MODELS'] or 0
    pct_drifted = (100 * drifted / total) if total > 0 else 0
    
    print(f"\nResults:")
    print(f"  • Predictions:     {pred_count:,} rows in IC_PREDICTIONS")
    print(f"  • Drifted models:  {drifted:,} ({pct_drifted:.1f}%)")
    print(f"  • Drift threshold: {drift_threshold}x baseline")
    
    if drifted > 0:
        print(f"\n⚠️ ACTION REQUIRED: Run train.py with MODE='SELECTIVE'")
    else:
        print(f"\n✅ Pipeline healthy - no action required")


def run_inference(session: Session = None):
    """Main entry point for inference and drift detection pipeline."""
    drift_threshold = infer_cfg["drift_threshold"]
    holdout_days = infer_cfg["holdout_days"]
    
    print(f"🔧 Inference Configuration:")
    print(f"   Drift threshold: {drift_threshold}x baseline")
    print(f"   Scoring window:  {holdout_days} days")
    
    if session is None:
        session = create_session("INFER")
        print(f"Connected: {session.get_current_account()}")
    
    latest_version = get_latest_model_version(session)
    train_run_id = f"ic_xgb_{latest_version}"
    inference_run_id = f"ic_inference_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
    
    print(f"📋 Using models from: {train_run_id}")
    print(f"   Inference run ID:  {inference_run_id}")
    
    prepare_scoring_data(session, holdout_days)
    execute_inference(session, train_run_id, inference_run_id)
    collect_predictions(session, inference_run_id)
    show_sample_predictions(session)
    run_drift_detection(session, drift_threshold, inference_run_id)
    drift_summary = get_drift_summary()
    print_drift_summary(drift_summary, drift_threshold)
    print_inference_summary(drift_summary, drift_threshold)


if __name__ == "__main__":
    run_inference()
