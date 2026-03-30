"""Many Model Training pipeline: trains one XGBoost model per partition."""
from snowflake.snowpark import Session
import snowflake.snowpark as sp
from snowflake.snowpark import functions as F
from snowflake.snowpark import Window
from snowflake.snowpark import types as T
from register import register_model

from xgboost import XGBRegressor
import pickle
import json

from datetime import datetime
import json
import tempfile
import os
from utils import (
    get_training_config, get_feature_config,
    create_session, get_stage_path,
    stage_data_partitioned, copy_from_stage_to_table
)

import pandas as pd
import numpy as np

train_cfg = get_training_config()
feature_cfg = get_feature_config()


GRAIN = feature_cfg["partition_col"]
TARGET = feature_cfg["target_col"]
TIME = feature_cfg["time_col"]
EXCLUDE_COLS = [GRAIN, TARGET, TIME]
TEST_PCT = feature_cfg["test_pct"]

def train_partition(df: pd.DataFrame) -> pd.DataFrame:
    """DPF worker: train XGBoost model, compute metrics, upload artifacts.
    Update this function for specific use cases"""
    
    if df.empty:
        return None
    
    df.columns = INPUT_COLS
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    X_train = df[feature_cols].astype("float32")
    y_train = df[TARGET].astype("float32")    
    
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_mae = float(np.mean(np.abs(y_train - train_pred)))
    train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
    
    feature_importances = {feat: float(imp) for feat, imp in zip(feature_cols, model.feature_importances_)}

    # Anything in this dictionary will be loaded into the model catalog table in a variant column
    metrics = {
        "TRAIN_MAE": train_mae,
        "TRAIN_RMSE": train_rmse,
        "TRAIN_ROWS": int(len(df)),
        "FEATURE_IMPORTANCES": feature_importances,
    }

    # Save the model
    model_binary = pickle.dumps(model)
    
    model_df = pd.DataFrame(
        [[model.__class__.__name__, model_binary, metrics]],
        columns=["ALGORITHM", "MODEL_BINARY", "METRICS"],
    )
    return model_df

def get_train_version() -> str:
    """Generate timestamped version string (e.g., v20240315_1430)."""
    return datetime.utcnow().strftime("v%Y%m%d_%H%M")

def prepare_data(session: Session):
    """Split data into train/test per partition, stage train data for DPF."""
    
    feature_table_name = feature_cfg["name"]
    test_pct = feature_cfg["test_pct"]
    
    features_df =  session.table(feature_table_name)
    
    partition_count = features_df.select(F.col(GRAIN)).distinct().count()
    print(f"🔄 Preparing data for {partition_count:,} partitions")
    print(f"   Test percentage: {test_pct * 100:.0f}%")
    
    window_spec = Window.partition_by(F.col(GRAIN)).order_by(F.col(TIME))
    features_with_rank = features_df.with_column(
        "ROW_NUM", F.row_number().over(window_spec)
    )
    
    partition_counts = features_df.group_by(F.col(GRAIN)).agg(
        F.count("*").alias("PARTITION_COUNT")
    )
    
    features_with_split = features_with_rank.join(
        partition_counts, on=GRAIN
    ).with_column(
        "TRAIN_CUTOFF", F.floor(F.col("PARTITION_COUNT") * F.lit(1 - test_pct))
    ).with_column(
        "IS_TRAIN", F.col("ROW_NUM") <= F.col("TRAIN_CUTOFF")
    )
    
    columns_to_keep = [c for c in features_df.columns]
    
    train_df = features_with_split.filter(F.col("IS_TRAIN")).select(columns_to_keep)
    test_df = features_with_split.filter(~F.col("IS_TRAIN")).select(columns_to_keep)

    train_count = train_df.count()
    test_count = test_df.count()
    
    train_df.write.mode("overwrite").save_as_table("TRAIN_DATA")
    test_df.write.mode("overwrite").save_as_table("TEST_DATA")

    
    print(f"   Train rows: {train_count:,}")
    print(f"   Test rows: {test_count:,}")
        
    return train_df, test_df

class ModelTrainingUDTF:
    """Class which is registered as a UDTF to train forecasting models."""

    def end_partition(self, df):
        """End partition method which utilizes the train model function."""
        model_df = train_partition(df)
        yield model_df


def execute_training(session: Session, train_run_id: str, train_df: sp.DataFrame):
    """Run ManyModelTraining to train models across all partitions."""
    global INPUT_COLS

    INPUT_COLS = [c for c in train_df.columns if c not in [TIME, GRAIN]]
    vect_udtf_input_dtypes = [
        T.PandasDataFrameType(
            [
                field.datatype
                for field in train_df.schema.fields
                if field.name in INPUT_COLS
            ]
        )
    ]

    udtf_name = f"MODEL_TRAINER_{train_run_id}"
    session.udtf.register(
        ModelTrainingUDTF,
        name=udtf_name,
        input_types=vect_udtf_input_dtypes,
        output_schema=T.PandasDataFrameType(
            [T.StringType(), T.BinaryType(), T.VariantType()],
            ["ALGORITHM", "MODEL_BINARY", "METRICS"],
        ),
        packages=[
            "snowflake-snowpark-python",
            "pandas",
            "numpy",
            "xgboost",
            "scikit-learn",
        ],
        replace=True,
        is_permanent=False,
    )

    udtf_models = train_df.select(
        GRAIN,
        TIME,
        TARGET,
        F.call_table_function(udtf_name, *INPUT_COLS).over(
            partition_by=[GRAIN],
            order_by=TIME,
        ),
    )

    udtf_models = udtf_models.select(
        GRAIN,
        "ALGORITHM",
        F.lit(train_run_id).alias("RUN_ID"),
        "MODEL_BINARY",
        "METRICS",
    )

    stage_path = get_stage_path()
    rows = udtf_models.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        for row in rows:
            partition_id = row[GRAIN]
            model_binary = row["MODEL_BINARY"]
            metrics = row["METRICS"]

            partition_dir = os.path.join(tmpdir, str(partition_id))
            os.makedirs(partition_dir, exist_ok=True)

            pkl_path = os.path.join(partition_dir, "model.pkl")
            with open(pkl_path, "wb") as f:
                f.write(model_binary)

            stage_dest = f"{stage_path}/{train_run_id}/{partition_id}/"
            session.file.put(pkl_path, stage_dest, auto_compress=False, overwrite=True)

    metrics_df = session.create_dataframe(
        [
            (row[GRAIN], datetime.utcnow().strftime("%Y-%m-%d"), row["METRICS"])
            for row in rows
        ],
        schema=["PARTITION_ID", "TRAINED_AT", "METRICS"],
    )
    metrics_df.write.mode("overwrite").save_as_table("MODEL_STAGING")

    return udtf_models


def collect_training_metrics(session: Session, train_run_id: str):
    """Load metrics parquet files from stage into MODEL_STAGING table."""
    session.sql("""
        CREATE OR REPLACE TEMPORARY TABLE MODEL_STAGING (
            PARTITION_ID VARCHAR(10),
            TRAINED_AT DATE,
            METRICS VARIANT
        );
    """).collect()
    copy_from_stage_to_table(session, "MODEL_STAGING", f"{train_run_id}/train_features", truncate_first=True)


def update_model_catalog(session: Session, train_version: str, train_run_id: str):
    """Merge new model versions and stage paths into MODEL_CATALOG."""
    print(f"\n📝 Updating MODEL_CATALOG...")
    
    stage_path = get_stage_path()
    fq_str = ".".join(stage_path.split(".")[:-1])
    metrics_df = session.table("MODEL_STAGING")
    catalog_df = metrics_df.with_columns(["MODEL_VERSION","IS_ACTIVE"], [F.lit(train_version),F.lit(True)])

    artifact_rows = session.sql(f"LIST '{stage_path}/{train_run_id}'").collect()

    subdir_to_model_path = []
    for row in artifact_rows:
        raw_name = row["name"]
        if raw_name.endswith(f"/model.pkl"):
            model_dir = raw_name.rsplit("/", 1)[0]
            parts = model_dir.split("/")
            subdir = parts[-1]
            subdir_to_model_path.append((subdir,f"{fq_str}.{model_dir}"))
    paths = session.create_dataframe(pd.DataFrame(subdir_to_model_path, columns = [GRAIN,"STAGE_PATH"]))

    print("catalog length:", catalog_df.count())

    catalog_df = catalog_df.join(paths, on = GRAIN)

    print("catalog length after merge:", catalog_df.count())

    catalog_df.write.mode("overwrite").save_as_table("MODEL_CATALOG")

    print(f"{catalog_df.count()} models saved to MODEL_CATALOG")



def run_training(session: Session):
    """Entry point: prepare data, train models, update catalog. Returns train_run_id."""
    
    train_version = get_train_version()
    train_run_id = f"training_{train_version}"

    session.sql(f"ALTER SESSION SET QUERY_TAG = '{train_run_id}'").collect()
    
    train_df, test_df = prepare_data(session)
    execute_training(session, train_run_id, train_df)
    update_model_catalog(session, train_version, train_run_id)

    return train_run_id

if __name__ == "__main__":
    session = create_session()
    print(f"Connected: {session.get_current_account()}")
    run_id = run_training(session)
    register_model(session)
    print(f"RUN_ID={run_id}")
