"""Utility functions for ML pipeline configuration and Snowflake operations."""
import yaml
from pathlib import Path
from datetime import date
from typing import Optional
from snowflake.snowpark import Session
import modin.pandas as pd
import snowflake.snowpark.modin.plugin

_config = None


def load_config(config_path: str = None) -> dict:
    """Load and cache configuration from YAML file."""
    global _config
    if _config is None:
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get_connection_config() -> dict:
    """Return connection settings from config."""
    return load_config()["connection"]


def get_tables_config() -> dict:
    """Return table name mappings from config."""
    return load_config()["tables"]


def get_training_config() -> dict:
    """Return training parameters from config."""
    return load_config()["ml_jobs"]["training"]


def get_inference_config() -> dict:
    """Return inference parameters from config."""
    return load_config()["ml_jobs"]["inference"]


def get_connection_config_legacy() -> dict:
    """Legacy alias for get_connection_config (backward compatibility)."""
    return get_connection_config()


def get_poc_data_config() -> dict:
    """Return POC data generation settings, parsing date string to date object."""
    cfg = load_config()["poc_data"]
    cfg["start_date"] = date.fromisoformat(cfg["start_date"])
    return cfg


def get_fully_qualified_name(table_key: str) -> str:
    """Build fully qualified table name from config key."""
    conn_cfg = get_connection_config()
    tables_cfg = get_tables_config()
    table_name = tables_cfg.get(table_key, table_key)
    return f"{conn_cfg['database']}.{conn_cfg['schema_data']}.{table_name}"


def get_stage_path(subpath: str = "") -> str:
    """Build fully qualified stage path with optional subpath suffix."""
    conn_cfg = get_connection_config()
    base = f"@{conn_cfg['database']}.{conn_cfg['schema_data']}.{conn_cfg['stage_artifacts']}"
    if subpath:
        return f"{base}/{subpath}"
    return base


def create_session(query_tag_suffix: str = "") -> Session:
    """Create and configure a Snowpark session using config settings."""
    conn_cfg = get_connection_config()
    session = Session.builder.getOrCreate()
    session.use_database(conn_cfg['database'])
    session.use_schema(conn_cfg['schema_data'])
    session.use_warehouse(conn_cfg['warehouse'])
    if conn_cfg['tag']:
        session.query_tag = conn_cfg['tag']+query_tag_suffix
    return session


def get_table(table_key: str) -> pd.DataFrame:
    """Return Snowpark DataFrame for the specified table config key."""
    fqn = get_fully_qualified_name(table_key)
    return pd.read_snowflake(fqn)


def get_max_date(table_key: str, date_col: str = "ACTUAL_DATE") -> Optional[date]:
    """Get maximum date value from a table, returning None if empty or missing."""
    try:
        df = get_table(table_key)
        if len(df) == 0:
            return None
        max_val = df[date_col].max()
        if pd.isna(max_val):
            return None
        return max_val.date() if hasattr(max_val, 'date') else max_val
    except Exception:
        return None


def stage_data_partitioned(session: Session, source_table_key: str, stage_subpath: str, 
                           partition_col: str = "STORE_MATERIAL_KEY"):
    """Export table to stage as partitioned parquet files for distributed processing."""
    stage_path = get_stage_path(stage_subpath)
    fqn = get_fully_qualified_name(source_table_key)
    
    session.sql(f"REMOVE {stage_path}/").collect()
    session.sql(f"""
        COPY INTO {stage_path}/
        FROM {fqn}
        PARTITION BY {partition_col}
        FILE_FORMAT = (TYPE = PARQUET COMPRESSION = SNAPPY)
        MAX_FILE_SIZE = 15728640
        HEADER = TRUE
    """).collect()
    session.sql(f"LIST {stage_path}/").show()


def copy_from_stage_to_table(session: Session, target_table_key: str, stage_subpath: str,
                              pattern: str = ".*[.]parquet", truncate_first: bool = True):
    """Load parquet files from stage into a table."""
    stage_path = get_stage_path(stage_subpath)
    tables_cfg = get_tables_config()
    table_name = tables_cfg.get(target_table_key, target_table_key)
    
    if truncate_first:
        session.sql(f"TRUNCATE TABLE IF EXISTS {table_name}").collect()
    
    session.sql(f"""
        COPY INTO {table_name}
        FROM {stage_path}/
        FILE_FORMAT = (TYPE = PARQUET COMPRESSION = SNAPPY)
        PATTERN = '{pattern}'
        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
    """).collect()


def group_values_by_delivery(df, grouping_col: str, sum_col: str):
    """Roll up values to delivery days by cumulative sum within groups."""
    import numpy as np
    df = df.copy()
    df["_GROUP"] = df[grouping_col][::-1].cumsum()[::-1]
    df[sum_col + "_GROUPED"] = df.groupby("_GROUP")[sum_col].cumsum()
    df[sum_col + "_GROUPED"] = np.where(
        df[grouping_col] == False, 0, df[sum_col + "_GROUPED"]
    )
    return df.drop(columns=["_GROUP"])
