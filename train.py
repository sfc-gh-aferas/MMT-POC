"""Distributed model training pipeline for IC volume forecasting."""
from snowflake.snowpark import Session
from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.modeling.distributors.many_model import ManyModelTraining, PickleSerde
from snowflake.ml.modeling.distributors.distributed_partition_function.entities import ExecutionOptions
from datetime import datetime
import json
from utils import (
    get_training_config, get_tables_config,
    create_session, get_fully_qualified_name, get_stage_path, get_table,
    stage_data_partitioned, copy_from_stage_to_table
)

import modin.pandas as pd
import snowflake.snowpark.modin.plugin

train_cfg = get_training_config()
tables_cfg = get_tables_config()


def get_train_version() -> str:
    """Generate timestamped version string."""
    version = datetime.utcnow().strftime("v%Y%m%d_%H%M")
    return version


GRAIN = ["MANAGED_BY", "SALES_OFFICE", "STORE_NUM", "MATERIAL_ID"]
TARGET = "WM_DAILY_CASE_VOLUME"
EXCLUDE_COLS = [
    "ACTUAL_DATE", "STORE_MATERIAL_KEY", "DELIVERY_PLAN",
    "WM_DAILY_VOLUME_EACH", "VOLUME", "ORDERS",
]


def train_xgb_partition(data_connector: DataConnector, context):
    """Train XGBoost model for one partition, returning None if insufficient data."""
    import pandas as pd
    import numpy as np
    import json
    from datetime import datetime
    from xgboost import XGBRegressor
    
    df = data_connector.to_pandas()
    if df.empty:
        return None
    
    df = df.sort_values("ACTUAL_DATE")
    cutoff = df["CUTOFF_DATE"].iloc[0]
    
    train = df[df["ACTUAL_DATE"] < cutoff]
    test = df[df["ACTUAL_DATE"] >= cutoff]
    
    if len(train) < 40 or len(test) < 7:
        return None
    
    exclude = set(GRAIN + EXCLUDE_COLS + [TARGET, "CUTOFF_DATE"])
    feature_cols = [c for c in df.columns if c not in exclude]
    
    y_train = train[TARGET].astype("float32")
    y_test = test[TARGET].astype("float32")
    
    base_score = int(y_train[y_train.values > 0].mean()) if (y_train > 0).any() else 0
    
    xgb_params = {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1, "random_state": 42,
        "base_score": base_score, "eval_metric": "mae",
    }
    
    X_train = train[feature_cols].astype("float32")
    X_test = test[feature_cols].astype("float32")
    
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mae = float(np.mean(np.abs(y_train - train_pred)))
    test_mae = float(np.mean(np.abs(y_test - test_pred)))
    train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
    test_rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))
    
    feature_importances = {feat: float(imp) for feat, imp in zip(feature_cols, model.feature_importances_)}
    
    metrics = pd.DataFrame([{
        "PARTITION_ID": context.partition_id.split('/')[1],
        "TRAIN_MAE": train_mae, "TEST_MAE": test_mae,
        "TRAIN_RMSE": train_rmse, "TEST_RMSE": test_rmse,
        "TRAIN_ROWS": int(len(train)), "TEST_ROWS": int(len(test)),
        "TRAINED_AT": datetime.utcnow().isoformat(),
        "FEATURE_IMPORTANCES": feature_importances,
        "HYPERPARAMETERS": xgb_params,
    }])
    
    context.upload_to_stage(metrics, "metrics.parquet",
                            write_function=lambda pdf, path: pdf.to_parquet(path, index=False))
    return model


def get_drifted_partitions() -> list:
    """Query MODEL_PERFORMANCE_LOG for partitions flagged as drifted today."""
    df = get_table("model_performance_log")
    df = df[(df["LOG_DATE"] == pd.Timestamp.today().normalize()) & (df["IS_DRIFTED"] == True)]
    return df["STORE_MATERIAL_KEY"].unique().tolist()


def prepare_training_data(session: Session):
    """Stage training data with CUTOFF_DATE for consistent train/test splits."""
    features_enriched = get_table("features_enriched")
    partition_count = features_enriched["STORE_MATERIAL_KEY"].nunique()
    print(f"🔄 FULL MODE: Training {partition_count:,} models")
    
    global_max_date = features_enriched["ACTUAL_DATE"].max()
    cutoff_date = global_max_date - pd.Timedelta(days=56)
    features_enriched["CUTOFF_DATE"] = cutoff_date
    
    print(f"   Training rows: {len(features_enriched):,}")
    print(f"   Global max date: {global_max_date}")
    print(f"   Train/test cutoff: {cutoff_date}")
    features_enriched.to_snowflake("TRAIN_DATA", if_exists="replace", index=False, table_type="temp")
    stage_data_partitioned(session, "train_data", "enriched_features")
    return features_enriched


def execute_training(session: Session, train_run_id: str):
    """Execute distributed training using ManyModelTraining framework."""
    stage_path = get_stage_path()
    trainer = ManyModelTraining(
        train_func=train_xgb_partition,
        stage_name=stage_path.replace("@", ""),
        serde=PickleSerde(),
    )
    
    train_run = trainer.run_from_stage(
        stage_location=f"{stage_path}/enriched_features/",
        run_id=train_run_id,
        file_pattern="*.parquet",
        on_existing_artifacts="overwrite",
        execution_options=ExecutionOptions(use_head_node=False, num_cpus_per_worker=1),
    )
    
    train_status = train_run.wait()
    print(f"\n   Training status: {train_status}")
    return train_status


def collect_training_metrics(session: Session, train_run_id: str):
    """Load training metrics from stage into TRAINING_METRICS table."""
    copy_from_stage_to_table(session, tables_cfg["training_metrics"], f"{train_run_id}/enriched_features", truncate_first=True)


def update_model_catalog(session: Session, train_version: str, train_run_id: str):
    """Update MODEL_CATALOG with new versions, archiving replaced models first."""
    print(f"\n📝 Updating MODEL_CATALOG...")
    
    stage_path = get_stage_path()
    metrics_df = get_table("training_metrics")

    artifact_rows = session.sql(f"LIST '{stage_path}/{train_run_id}'").collect()

    subdir_to_model_path = []
    for row in artifact_rows:
        raw_name = row["name"]
        if raw_name.endswith(f"/model.pkl"):
            model_dir = raw_name.rsplit("/", 1)[0]
            parts = model_dir.split("/")
            subdir = parts[-2]
            subdir_to_model_path.append((subdir,f"@{model_dir}"))
    paths = pd.DataFrame(subdir_to_model_path, columns = ["STORE_MATERIAL_KEY","STAGE_PATH"])
    
    catalog_df = pd.DataFrame({
        "STORE_MATERIAL_KEY": metrics_df["PARTITION_ID"],
        "MODEL_VERSION": train_version,
        "TRAINED_AT": metrics_df["TRAINED_AT"],
        "TRAIN_ROWS": metrics_df["TRAIN_ROWS"],
        "TRAIN_MAE": metrics_df["TRAIN_MAE"],
        "TEST_MAE": metrics_df["TEST_MAE"],
        "TRAIN_RMSE": metrics_df["TRAIN_RMSE"],
        "TEST_RMSE": metrics_df["TEST_RMSE"],
        "IS_ACTIVE": True,
        "FEATURE_IMPORTANCES": metrics_df["FEATURE_IMPORTANCES"].apply(lambda x: json.dumps(x)),
        "HYPERPARAMETERS": metrics_df["HYPERPARAMETERS"].apply(lambda x: json.dumps(x)),
    })

    print("catalog length:", len(catalog_df))

    catalog_df = catalog_df.merge(paths, on = "STORE_MATERIAL_KEY")

    print("catalog length after merge:", len(catalog_df))
    
    if len(catalog_df) > 0:
        partition_keys = catalog_df["STORE_MATERIAL_KEY"].unique().tolist()
        partition_keys_sql = ",".join([f"'{k}'" for k in partition_keys])
        
        catalog_fqn = get_fully_qualified_name("model_catalog")
        history_fqn = get_fully_qualified_name("model_catalog_history")
        
        session.sql(f"""
            INSERT INTO {history_fqn}
            SELECT STORE_MATERIAL_KEY, MODEL_VERSION, STAGE_PATH, TRAINED_AT,
                CURRENT_TIMESTAMP(), TRAIN_MAE, TEST_MAE, 'Replaced by {train_version}'
            FROM {catalog_fqn}
            WHERE STORE_MATERIAL_KEY IN ({partition_keys_sql}) AND IS_ACTIVE = TRUE
        """).collect()
        
        catalog_df.to_snowflake("MODEL_CATALOG_STAGING", if_exists="replace", index=False)
        
        session.sql(f"""
            MERGE INTO {catalog_fqn} t USING MODEL_CATALOG_STAGING s
            ON t.STORE_MATERIAL_KEY = s.STORE_MATERIAL_KEY
            WHEN MATCHED THEN UPDATE SET
                MODEL_VERSION = s.MODEL_VERSION, STAGE_PATH = s.STAGE_PATH,
                TRAINED_AT = s.TRAINED_AT, TRAIN_ROWS = s.TRAIN_ROWS,
                TRAIN_MAE = s.TRAIN_MAE, TEST_MAE = s.TEST_MAE,
                TRAIN_RMSE = s.TRAIN_RMSE, TEST_RMSE = s.TEST_RMSE,
                IS_ACTIVE = TRUE,
                FEATURE_IMPORTANCES = PARSE_JSON(s.FEATURE_IMPORTANCES),
                HYPERPARAMETERS = PARSE_JSON(s.HYPERPARAMETERS)
            WHEN NOT MATCHED THEN INSERT VALUES (
                s.STORE_MATERIAL_KEY, s.MODEL_VERSION, s.STAGE_PATH, s.TRAINED_AT,
                s.TRAIN_ROWS, s.TRAIN_MAE, s.TEST_MAE, s.TRAIN_RMSE, s.TEST_RMSE,
                TRUE, PARSE_JSON(s.FEATURE_IMPORTANCES), PARSE_JSON(s.HYPERPARAMETERS)
            )
        """).collect()
        
        session.sql("DROP TABLE IF EXISTS MODEL_CATALOG_STAGING").collect()


def print_training_summary(train_version: str):
    """Print final training summary with model catalog statistics."""
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print("=" * 60)
    
    catalog = get_table("model_catalog")
    active = catalog[catalog["IS_ACTIVE"] == True]
    
    active_models = len(active)
    versions = active["MODEL_VERSION"].nunique()
    avg_mae = active["TEST_MAE"].mean()
    
    print(f"\nMODEL_CATALOG Summary:")
    print(f"  • Active models:     {active_models:,}")
    print(f"  • Model versions:    {versions}")
    print(f"  • Avg baseline MAE:  {avg_mae:.4f}")
    print(f"  • Latest version:    {train_version}")


def run_training(session: Session = None):
    """Main entry point for model training pipeline."""
    
    if session is None:
        session = create_session("TRAIN")
        print(f"Connected: {session.get_current_account()}")
    
    train_version = get_train_version()
    train_run_id = f"ic_xgb_{train_version}"
    
    prepare_training_data(session)
    execute_training(session, train_run_id)
    collect_training_metrics(session, train_run_id)
    update_model_catalog(session, train_version, train_run_id)
    print_training_summary(train_version)


if __name__ == "__main__":
    run_training()
