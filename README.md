# MMT-POC: Many Model Training Proof of Concept/Consumption estimation

Proof of concept and consumption estimation for distributed many-model training and inference using Snowflake ML Jobs. Trains timeseries forecasting XGBoost models per partition using the Many Model Training (MMT) frameworks.

Uses notebooks for setup and orchestration to enable running either locally or in Snowflake workspaces.

**This poc will set up new databases, warehouses, and compute pools, but relies on an existing role and set of privileges from your default connection**

## Quick Start

### 1. Update Configuration

Edit `poc/config.yaml`
   - Data generation will follow the specifications set in poc_data. Choose the number of partitions/models, features, timesteps, and start date.
   - Set poc_data.generate to False if you already have your data prepared and do not want to generate it.
     You will need to update the feature_table configs to ensure they match your data and use case
   - Update training and inference compute configurations.

### 2. Edit Training/Inference Functions (Optional)

Customize the model training logic in `poc/train.py`:

```python
def train_partition(data_connector: DataConnector, context):
    # Modify model type, hyperparameters, feature engineering
    model = XGBRegressor()  # Change model here
    model.fit(X_train, y_train)
    return model
```

Customize inference logic in `poc/infer.py`:

```python
def predict_partition(data_connector: DataConnector, context):
    # Modify prediction logic, post-processing
    preds = model.predict(X)
    # Add custom transformations here
```

### 3. Run setup.ipynb

Creates infrastructure and generates POC data:
- Database, schema, and stage
- Warehouse configuration
- Synthetic feature table (if `poc_data.generate: True`)

### 4. Run submit_ml_jobs.ipynb

Submits ML Jobs to Snowflake compute pools:
- **Cell 1**: Submits training job (`poc/train.py`)
- **Cell 2**: Submits inference job (`poc/infer.py`)

Each job runs distributed across the configured cluster size.

## Project Structure

```
MMT-POC/
├── poc/
│   ├── config.yaml    # Central configuration
│   ├── utils.py       # Snowflake utilities
│   ├── train.py       # MMT training pipeline
│   └── infer.py       # DPF inference pipeline
├── setup.ipynb        # Infrastructure setup
├── submit_ml_jobs.ipynb # Job submission
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
