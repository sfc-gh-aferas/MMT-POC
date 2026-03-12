# MMT-POC: Distributed ML Pipeline for IC Volume Forecasting

**Proof of Concept** for testing Snowflake ML consumption patterns at scale. This pipeline forecasts IC volume by training thousands of XGBoost models in parallel using Snowflake ML's Distributed Partition Function (DPF) and Many Model Training frameworks.

## POC Scale & Consumption

Tested configuration: **700 stores × 20 materials × 2 years** = 10.2M rows, 14,000 models

| Stage | Runtime | Warehouse Credits | Compute Credits |
|-------|---------|-------------------|------------------|
| Features | 19 min | 0.15 | 1.86 |
| Training | 23 min | 0.15 | 2.25 |
| Inference | 15 min | 0.15 | 1.46 |
| **Total** | **~1 hour** | **0.45** | **5.57** |

**Infrastructure**: 50-node CPU_X64_S compute pool, Medium warehouse

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   00_setup      │ ──► │  01_poc_data    │ ──► │  orchestrate.py │
│  (Infrastructure)│     │ (Synthetic Data)│     │ (Task Creation) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                    Creates Snowflake Tasks (DAGs)
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        ▼                                ▼                                ▼
               ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
               │  IC_FEATURES    │ ──────────►  │    IC_TRAIN     │ ──────────►  │    IC_INFER     │
               │   (ML Job)      │              │    (ML Job)     │              │    (ML Job)     │
               └─────────────────┘              └─────────────────┘              └─────────────────┘
                 features.py                      train.py                         infer.py
```

## Features

- **Distributed Feature Engineering**: Time-series features (lags, rolling stats, seasonality) computed per partition
- **Many-Model Training**: One XGBoost model per store-material combination (~14,000 models)
- **Model Governance**: Versioned model catalog with automatic archiving
- **Drift Detection**: Monitors prediction accuracy and flags models for retraining
- **Holiday-Aware**: Lead-in/lead-out holiday effects with per-holiday dummy variables

## Quick Start

### Prerequisites

- Snowflake account
- Python 3.12+ with conda environment
- Snowflake CLI configured

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `ml_jobs/config.yaml` to set:
- Database/schema names
- Compute pool settings
- Training hyperparameters
- POC data scale

### Running the Pipeline

1. **Setup Infrastructure** (one-time):
   ```bash
   # Run 00_setup.ipynb to create database, schemas, tables, and compute pool
   ```

2. **Generate POC Data** (optional):
   ```bash
   # Run 01_poc_data.ipynb to create synthetic test data
   ```

3. **Deploy ML Pipeline Tasks**:
   ```bash
   python orchestrate.py  # Creates Snowflake Tasks for each ML job
   ```

4. **Run Tasks**:
   ```sql
   -- Run individual tasks
   EXECUTE TASK IC_FEATURES;
   EXECUTE TASK IC_TRAIN;
   EXECUTE TASK IC_INFER;
   ```

## Project Structure

```
MMT_POC/
├── ml_jobs/
│   ├── config.yaml      # Central configuration
│   ├── utils.py         # Snowflake session & table utilities
│   ├── features.py      # Distributed feature engineering (DPF)
│   ├── train.py         # Many-model training pipeline
│   └── infer.py         # Inference & drift detection
├── 00_setup.ipynb       # Infrastructure setup
├── 01_poc_data.ipynb    # Synthetic data generation
├── orchestrate.py       # Task creation & ML job submission
└── requirements.txt     # Python dependencies
```

## Orchestration

The `orchestrate.py` script:
1. Uploads `ml_jobs/` directory to a Snowflake stage
2. Creates three Snowflake Tasks (DAGs) that submit ML Jobs:
   - `IC_FEATURES` - Runs `features.py` via `submit_from_stage`
   - `IC_TRAIN` - Runs `train.py` via `submit_from_stage`
   - `IC_INFER` - Runs `infer.py` via `submit_from_stage`

Each task uses the configured compute pool to run distributed workloads.

> **Note**: Task scheduling (CRON) is not currently configured. Tasks must be executed manually. Automated daily scheduling should be added in a future release.

## Key Tables

| Table | Purpose |
|-------|---------|
| `FACT_IC_VOLUME` | Raw daily case volume data |
| `DIM_CALENDAR` | Calendar with holidays |
| `FEATURES_ENRICHED` | Engineered features for ML |
| `MODEL_CATALOG` | Active model registry |
| `IC_PREDICTIONS` | Inference output |
| `MODEL_PERFORMANCE_LOG` | Drift metrics |

## DataFrame Processing

The pipeline uses two distinct pandas implementations:

| Context | Library | Reason |
|---------|---------|--------|
| Outside DPF (train.py, infer.py) | Snowpark pandas (`modin.pandas`) | Pushes computation to Snowflake warehouse for large-scale data prep |
| Inside DPF workers (features.py) | OSS pandas | Runs on compute pool nodes; Snowpark pandas not available in DPF context |

This hybrid approach leverages Snowflake's distributed compute for data preparation while using standard pandas for per-partition transformations within ML Jobs.

## Feature Engineering

Features computed per store-material partition:

- **Lag Features**: 7d, 14d, 28d, 56d, 112d, 168d, 224d, 365d
- **Rolling Stats**: 7-day mean/std, 365-day mean
- **Cyclical Encodings**: sin/cos for day, week, month
- **Seasonality**: Decomposed seasonal components at multiple periods
- **Holiday Effects**: Per-holiday indicators with lead-in/lead-out windows
- **YoY Comparisons**: Year-over-year daily and day-of-week averages

## Model Training

- **Algorithm**: XGBoost Regressor
- **Target**: Log-transformed case volume
- **Train/Test Split**: 56-day holdout
- **Hyperparameters**: Configured in `config.yaml`
- **Output**: Pickle files staged per partition

## Drift Detection

After inference, the pipeline:
1. Computes rolling MAE over 7/14-day windows
2. Compares to baseline test MAE from training
3. Flags models where `drift_ratio > threshold` (default: 1.5x)
4. Logs results to `MODEL_PERFORMANCE_LOG`

## Future Considerations

This POC validates the core ML pipeline architecture. Areas for future expansion include: automated task scheduling via CRON for daily pipeline execution, integration with Snowflake Model Registry for centralized model versioning and lifecycle management, leveraging Snowflake Feature Store for reusable feature definitions across projects, and deploying models to Snowflake endpoints for real-time inference. Additionally, the feature engineering stage could be optimized by replacing DPF with pure Snowpark transformations or UDTFs to reduce compute pool usage.

## Dependencies

Key packages:
- `snowflake-ml-python` - Distributed ML framework
- `snowflake-snowpark-python` - DataFrame API
- `xgboost` - Model training
- `pandas`, `numpy`, `scipy` - Data processing
- `statsmodels` - Seasonal decomposition

