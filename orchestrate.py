"""Orchestration script for ML pipeline using Snowflake Tasks."""
import os
import yaml
from pathlib import Path
from datetime import timedelta

from snowflake.snowpark import Session
from snowflake.core import Root, CreateMode
from snowflake.core.task.dagv1 import DAG, DAGTask, DAGOperation
from snowflake.core.task.context import TaskContext
from snowflake.ml.jobs import submit_from_stage


from ml_jobs.utils import get_connection_config, get_stage_path, create_session

def upload_ml_jobs_to_stage(session: Session, stage_name: str) -> str:
    """Upload ml_jobs subdirectory to stage, returning the stage path."""
    ml_jobs_dir = Path(__file__).parent / "ml_jobs"
    stage_path = f"@{stage_name}/ml_jobs"
    
    session.sql(f"REMOVE {stage_path}/").collect()
    
    files_to_upload = [
        "config.yaml",
        "utils.py",
        "features.py",
        "train.py",
        "infer.py",
    ]
    
    for filename in files_to_upload:
        local_path = ml_jobs_dir / filename
        if local_path.exists():
            session.file.put(
                str(local_path),
                f"{stage_path}/",
                auto_compress=False,
                overwrite=True
            )
            print(f"  Uploaded: {filename}")
    
    print(f"\nFiles uploaded to {stage_path}/")
    session.sql(f"LIST {stage_path}/").show()
    return stage_path

def get_mljob_runner(stage_name:str, source:str, entrypoint:str, compute_pool:str, nodes:int):
    """Create a task function that submits and runs a Python script as a Snowflake ML Job."""
    def job_func(session: Session) ->  str:
        ctx = TaskContext(session)
        job = submit_from_stage(
            source=source,
            compute_pool=compute_pool,
            entrypoint=entrypoint,
            stage_name=stage_name,
            target_instances=nodes,
            session=session,
        )
        job.wait()
        return job.result()
    return job_func


def create_mljobs_dags(session: Session, stage_path:str, compute_pool:str, nodes:int):
    """Create DAGTasks that call submit_from_stage for each ML job."""
        
    database = session.get_current_database()
    schema = session.get_current_schema()
    warehouse = session.get_current_warehouse()
    
    root = Root(session)
    schema_ref = root.databases[database].schemas[schema]
    dag_op = DAGOperation(schema_ref)
    stage_name = "/".join(stage_path.split("/")[:-1])
    for name in ["features","train","infer"]:
        task_name = f"IC_{name.upper()}"
        with DAG(
            name=task_name,
            warehouse=warehouse, 
            stage_location=stage_path,
            packages=["snowflake-snowpark-python","snowflake-ml-python"]
        ) as dag:        
            task_func = get_mljob_runner(stage_name=stage_name, source=stage_path, entrypoint=f"{name}.py", compute_pool=compute_pool, nodes=nodes)
            task = DAGTask(
                name=f"{name}_mljob",
                definition=task_func,
            )

        dag_op.deploy(dag, mode=CreateMode.or_replace)
        print(f"Task {task_name} deployed!")


def main():
    """Main entry point for orchestration setup."""
    print("=" * 60)
    print("ML PIPELINE ORCHESTRATION SETUP")
    print("=" * 60)
    
    conn_cfg = get_connection_config()
    
    session = create_session()
    
    print(f"\nConnected to: {session.get_current_account()}")
    print(f"Database: {conn_cfg['database']}")
    print(f"Schema: {conn_cfg['schema_data']}")
    print(f"Compute Pool: {conn_cfg['compute_pool']}")
    
    print("\n" + "-" * 60)
    print("Step 1: Uploading ml_jobs to stage...")
    print("-" * 60)
    
    stage_name = f"{conn_cfg['database']}.{conn_cfg['schema_data']}.{conn_cfg['stage_artifacts']}"
    stage_path = upload_ml_jobs_to_stage(session, stage_name)
    
    print("\n" + "-" * 60)
    print("Step 2: Creating ML Pipeline DAG...")
    print("-" * 60)
    
    create_mljobs_dags(session=session, stage_path=stage_path, compute_pool=conn_cfg["compute_pool"], nodes=conn_cfg["target_cluster_size"])
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Resume the root task to enable scheduling:")
    print(f"     ALTER TASK {conn_cfg['database']}.{conn_cfg['schema_data']}.ML_PIPELINE_DAG RESUME;")
    print("  2. Or run immediately:")
    print(f"     EXECUTE TASK {conn_cfg['database']}.{conn_cfg['schema_data']}.ML_PIPELINE_DAG;")
    print("  3. Monitor task history:")
    print(f"     SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY()) ORDER BY SCHEDULED_TIME DESC;")


if __name__ == "__main__":
    main()
