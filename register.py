"""Model registration pipeline for IC volume forecasting."""
import pickle
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from snowflake.ml.model import custom_model
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session
from utils import (
    get_connection_config, get_tables_config,
    create_session, get_fully_qualified_name
)

conn_cfg = get_connection_config()
tables_cfg = get_tables_config()

GRAIN = ["MANAGED_BY", "SALES_OFFICE", "STORE_NUM", "MATERIAL_ID"]
TARGET = "WM_DAILY_CASE_VOLUME"
EXCLUDE_COLS = [
    "ACTUAL_DATE", "STORE_MATERIAL_KEY", "DELIVERY_PLAN",
    "WM_DAILY_VOLUME_EACH", "VOLUME", "ORDERS", "CUTOFF_DATE",
]


class ICVolumeForecastModel(custom_model.CustomModel):
    """Partitioned CustomModel using @partitioned_api for distributed inference."""
    
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        self._model_cache: Dict[str, Any] = {}
    
    def _get_model(self, partition_key: str) -> Any:
        """Load model for a partition from stage, with caching."""
        if partition_key not in self._model_cache:
            stage_path = self.context[partition_key]
            if stage_path is None:
                return None
            
            from snowflake.snowpark.files import SnowflakeFile
            with SnowflakeFile.open(stage_path, "rb") as f:
                self._model_cache[partition_key] = pickle.load(f)
        
        return self._model_cache.get(partition_key)
    
    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for a single partition."""
        import numpy as np
        
        partition_key = input_df["STORE_MATERIAL_KEY"].iloc[0]
        
        model = self._get_model(partition_key)
        if model is None:
            return pd.DataFrame({
                "OUTPUT_STORE_MATERIAL_KEY": input_df["STORE_MATERIAL_KEY"],
                "PRED_LOG_TARGET": [None] * len(input_df),
                "PRED_CASES": [None] * len(input_df),
            })
        
        exclude = set(GRAIN + EXCLUDE_COLS + [TARGET, "PRED_LOG_TARGET", "PRED_CASES"])
        feature_cols = [c for c in input_df.columns if c not in exclude]
        X = input_df[feature_cols].astype("float32")
        log_preds = model.predict(X)
        pred_cases = np.maximum(0, (10 ** log_preds) - 1)
        
        return pd.DataFrame({
            "OUTPUT_STORE_MATERIAL_KEY": input_df["STORE_MATERIAL_KEY"].values,
            "PRED_LOG_TARGET": log_preds,
            "PRED_CASES": pred_cases,
        })


def create_sample_input(session: Session) -> pd.DataFrame:
    """Sample from FEATURES_ENRICHED table for schema inference."""
    features_fqn = get_fully_qualified_name("features_enriched")
    
    sample_df = session.sql(f"""
        SELECT *
        FROM {features_fqn}
        LIMIT 100
    """).to_pandas()
    
    exclude = set(GRAIN + EXCLUDE_COLS + [TARGET])
    keep_cols = ["STORE_MATERIAL_KEY"] + [c for c in sample_df.columns if c not in exclude and c != "STORE_MATERIAL_KEY"]
    return sample_df[keep_cols]


def build_stage_paths(session: Session) -> Dict[str, str]:
    """Build mapping of partition key -> stage path from MODEL_CATALOG."""
    catalog_fqn = f"{conn_cfg['database']}.{conn_cfg['schema_data']}.{tables_cfg['model_catalog']}"
    catalog_df = session.sql(f"""
        SELECT STORE_MATERIAL_KEY, STAGE_PATH 
        FROM {catalog_fqn} 
        WHERE IS_ACTIVE = TRUE
    """).to_pandas()
    print(catalog_df.head(5))
    
    return dict(zip(catalog_df["STORE_MATERIAL_KEY"], catalog_df["STAGE_PATH"].str.replace('@','@MMT_POC.FORECASTING.')))


def register_model(session: Session = None, registry_name: str = "IC_VOLUME_FORECAST"):
    """Register ManyModelTraining models to Snowflake Model Registry."""
    if session is None:
        session = create_session("REGISTER")
        print(f"Connected: {session.get_current_account()}")
    
    print(f"\n📦 Registering Partitioned Model to Snowflake Model Registry")
    print(f"   Registry name: {registry_name}")
    
    reg = Registry(
        session=session,
        database_name=conn_cfg["database"],
        schema_name=conn_cfg["schema_data"]
    )
    
    stage_paths = build_stage_paths(session)
    print(f"   Found {len(stage_paths)} active models")

    
    model_context = custom_model.ModelContext(
        artifacts=stage_paths
    )
    wrapper = ICVolumeForecastModel(model_context)
    
    sample_input = create_sample_input(session)
    print(f"   Sample input shape: {sample_input.shape}")
    
    print(f"\n🚀 Logging partitioned model to registry...")
    
    mv = reg.log_model(
        wrapper,
        model_name=registry_name,
        sample_input_data=sample_input,
        options={"function_type": "TABLE_FUNCTION"},
        conda_dependencies=["xgboost==2.0.3", "pandas", "numpy==1.26.4"],
        target_platforms=["WAREHOUSE"],
        comment=f"IC Volume Forecast Partitioned Model"
    )
    
    model = reg.get_model(registry_name)
    model.default = mv.version_name
    
    print(f"\n✅ Model registered successfully!")
    print(f"   Model: {mv.model_name}")
    print(f"   Version: {mv.version_name} (set as default)")
    print(f"   Database: {conn_cfg['database']}")
    print(f"   Schema: {conn_cfg['schema_data']}")
    
    print(f"\n📋 Available methods:")
    print(mv.show_functions())
    
    return mv


if __name__ == "__main__":
    
    register_model()
