# MMT-POC: Many Model Training Proof of Concept/Consumption estimation

Proof of concept and consumption estimation for distributed many-model training and inference using Snowflake. Trains timeseries forecasting XGBoost models per partition using the Many Model Training (MMT) frameworks. Supports running on either **ML Jobs (containers)** or **warehouses**, with optional model registration to the **Snowflake Model Registry**.

Uses notebooks for setup and orchestration to enable running either locally or in Snowflake workspaces.

**This poc will set up new databases, warehouses, and compute pools, but relies on an existing role and set of privileges from your default connection**

## Quick Start

### 1. Update Configuration

Edit `poc/config.yaml`
   - Data generation will follow the specifications set in poc_data. Choose the number of partitions/models, features, timesteps, and start date.
   - Set poc_data.generate to False if you already have your data prepared and do not want to generate it.
     You will need to update the feature_table configs to ensure they match your data and use case
   - Update training and inference compute configurations.
   - Set `training.mljob` / `inference.mljob` to `True` for ML Job (container) execution, or `False` for warehouse execution.
   - When using ML Jobs, specify `instance_family` and `target_cluster_size`.

### 2. Edit Training/Inference Functions (Optional)

**Container (ML Jobs):** Customize `poc/train.py` and `poc/infer.py` (DPF-based, distributed):

```python
def train_partition(data_connector: DataConnector, context):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model
```

**Warehouse:** Customize `poc/train_warehouse.py` and `poc/infer_warehouse.py` (UDTF-based, runs on warehouse):
- Training registers a UDTF and trains models per partition via SQL
- Inference uses the Snowflake Model Registry to run predictions

**Model Registration:** `poc/register.py` logs trained models to the Snowflake Model Registry as a `CustomModel` with `@partitioned_api`, enabling warehouse-based inference via `model.run()`.

### 3. Run setup.ipynb

Creates infrastructure and generates POC data:
- Database, schema, and stage
- Warehouse configuration
- Synthetic feature table (if `poc_data.generate: True`)

### 4. Run submit_ml_jobs.ipynb

Executes training and inference based on config:
- **Cell 1**: Runs training — ML Job (`poc/train.py`) or warehouse (`poc/train_warehouse.py`) based on `training.mljob`
- **Cell 2**: Runs inference — ML Job (`poc/infer.py`) or warehouse (`poc/infer_warehouse.py`) based on `inference.mljob`

Container jobs run distributed across the configured cluster size. Warehouse jobs execute via UDTFs and Model Registry.

## Project Structure

```
MMT-POC/
├── poc/
│   ├── config.yaml         # Central configuration
│   ├── utils.py            # Snowflake utilities
│   ├── train.py            # MMT training pipeline (container/ML Jobs)
│   ├── train_warehouse.py  # UDTF-based training pipeline (warehouse)
│   ├── infer.py            # DPF inference pipeline (container/ML Jobs)
│   ├── infer_warehouse.py  # Registry-based inference (warehouse)
│   └── register.py         # Model Registry registration
├── setup.ipynb             # Infrastructure setup
├── submit_ml_jobs.ipynb    # Job submission
└── requirements.txt
```

## Output Tables

| Table | Description |
|-------|-------------|
| `FEATURE_TABLE` | Input features (partitioned) |
| `TRAIN_DATA` | Training split |
| `TEST_DATA` | Test split for inference |
| `MODEL_CATALOG` | Trained model registry |
| `PREDICTIONS` | Inference results |
