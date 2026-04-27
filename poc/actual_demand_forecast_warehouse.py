"""Warehouse UDTF pipeline for weekly demand forecasting (mirrors actual_demand_forecast_spcs logic)."""
import snowflake.snowpark as sp
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark import Window
from snowflake.snowpark import types as T

import pandas as pd
import numpy as np
import re as _re
import gc
import time 
from datetime import datetime

"""Many Model Training pipeline for weekly demand forecasting via ML Jobs on SPCS."""
import pandas as pd
import numpy as np
import os
import random
import io
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark import Window
import utils
from utils import (
    get_training_config, get_feature_config,
    create_session, get_stage_path,
    stage_data_partitioned, copy_from_stage_to_table,
    get_fully_qualified_name,
)

import re as _re

# === DETERMINISTIC BEHAVIOR SETUP ===
# Set all random seeds for reproducible results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set CUDA path if not already set - adjust path as needed for your system
if 'CUDA_PATH' not in os.environ:
    # Try to auto-detect CUDA installation
    cuda_paths = [
        # Standard NVIDIA CUDA Toolkit installations
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6', 
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
        # CuPy bundled CUDA (pip/conda installations)
        r'C:\Users\ffr\python311\Lib\site-packages\cupy',
        # Generic patterns for different user installations
        os.path.expanduser(r'~\python311\Lib\site-packages\cupy'),
        os.path.expanduser(r'~\AppData\Local\Programs\Python\Python311\Lib\site-packages\cupy'),
    ]
    
    # Also try to detect from CuPy itself
    try:
        import cupy as cp_detect
        cupy_path = os.path.dirname(cp_detect.__file__)
        cuda_paths.insert(0, cupy_path)  # Prioritize detected CuPy path
        print(f"Detected CuPy installation at: {cupy_path}")
    except ImportError:
        pass
    
    for path in cuda_paths:
        if os.path.exists(path):
            os.environ['CUDA_PATH'] = path
            print(f"Auto-detected CUDA path: {path}")
            break
    else:
        print("No CUDA path auto-detected - CuPy will try default detection")



# imports not used
# import xlsxwriter
# from openpyxl import load_workbook

train_cfg = get_training_config()
feature_cfg = get_feature_config()

GRAIN = "ITEM_NUMBER_SITE"
TARGET = "ACTUAL_DEMAND"
TIME = "PERFORM_DATE"

def detect_structural_breaks(demand_data, date_col='PERFORM_DATE', demand_col='ACTUAL_DEMAND', 
                           min_break_size=90, cusum_threshold=2.5, variance_threshold=0.7):
    """
    Detect structural breaks in demand time series using multiple methods.
    
    Returns:
    - break_points: List of detected break dates
    - break_strength: Strength of strongest break (0-1 scale)
    - post_break_period: Data after the most recent significant break
    """
    if len(demand_data) < min_break_size * 2:
        return [], 0.0, demand_data
    
    ts_data = demand_data.copy().sort_values(date_col)
    demand_values = ts_data[demand_col].values
    dates = ts_data[date_col].values
    
    # Method 1: CUSUM-based break detection
    def cusum_break_detection(series, threshold=cusum_threshold):
        """Detect breaks using CUSUM statistics"""
        n = len(series)
        mean_series = np.mean(series)
        std_series = np.std(series)
        
        if std_series == 0:
            return []
        
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        break_candidates = []
        
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (series[i] - mean_series) / std_series)
            cusum_neg[i] = min(0, cusum_neg[i-1] + (series[i] - mean_series) / std_series)
            
            # Check for break
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                if i > min_break_size and i < n - min_break_size:
                    break_candidates.append((i, max(abs(cusum_pos[i]), abs(cusum_neg[i]))))
        
        return break_candidates
    
    # Method 2: Variance-based break detection
    def variance_break_detection(series, threshold=variance_threshold):
        """Detect breaks using variance ratio test"""
        n = len(series)
        break_candidates = []
        
        for i in range(min_break_size, n - min_break_size):
            pre_break = series[:i]
            post_break = series[i:]
            
            var_pre = np.var(pre_break) if len(pre_break) > 1 else 1.0
            var_post = np.var(post_break) if len(post_break) > 1 else 1.0
            
            if var_pre > 0 and var_post > 0:
                var_ratio = max(var_pre, var_post) / min(var_pre, var_post)
                if var_ratio > (1 / threshold):
                    break_candidates.append((i, var_ratio))
        
        return break_candidates
    
    # Method 3: Mean shift detection
    def mean_shift_detection(series):
        """Detect significant mean shifts"""
        n = len(series)
        break_candidates = []
        
        for i in range(min_break_size, n - min_break_size):
            pre_break = series[:i]
            post_break = series[i:]
            
            mean_pre = np.mean(pre_break)
            mean_post = np.mean(post_break)
            std_pre = np.std(pre_break) if len(pre_break) > 1 else 1.0
            std_post = np.std(post_break) if len(post_break) > 1 else 1.0
            
            # Calculate standardized mean difference
            pooled_std = np.sqrt((std_pre**2 + std_post**2) / 2)
            if pooled_std > 0:
                mean_diff = abs(mean_pre - mean_post) / pooled_std
                if mean_diff > 1.5:  # Significant shift threshold
                    break_candidates.append((i, mean_diff))
        
        return break_candidates
    
    cusum_breaks = cusum_break_detection(demand_values)
    variance_breaks = variance_break_detection(demand_values)
    mean_breaks = mean_shift_detection(demand_values)
    
    all_breaks = {}
    
    # Weight different methods
    for idx, strength in cusum_breaks:
        all_breaks[idx] = all_breaks.get(idx, 0) + strength * 0.4
    
    for idx, strength in variance_breaks:
        all_breaks[idx] = all_breaks.get(idx, 0) + (strength - 1) * 0.3
    
    for idx, strength in mean_breaks:
        all_breaks[idx] = all_breaks.get(idx, 0) + strength * 0.3
    
    # Filter and sort breaks by strength
    significant_breaks = [(idx, strength) for idx, strength in all_breaks.items() 
                         if strength > 1.0]  # Combined threshold
    significant_breaks.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to dates and return results
    break_points = [dates[idx] for idx, _ in significant_breaks[:3]]  # Top 3 breaks
    break_strength = significant_breaks[0][1] if significant_breaks else 0.0
    
    # Identify post-break period (after most recent significant break)
    if break_points:
        most_recent_break = max(break_points)
        post_break_mask = ts_data[date_col] > most_recent_break
        post_break_period = ts_data[post_break_mask].copy()
    else:
        post_break_period = ts_data.copy()
    
    # Normalize break strength to 0-1 scale
    break_strength = min(break_strength / 5.0, 1.0)
    
    return break_points, break_strength, post_break_period

def ensemble_forecast_with_breaks(historical_data, post_break_data, break_strength,
                                date_col='PERFORM_DATE', demand_col='ACTUAL_DEMAND',
                                forecast_horizon=30):
    """
    Generate ensemble forecast considering structural breaks.
    
    Returns:
    - ensemble_forecast: Combined forecast
    - confidence_multiplier: Factor to expand confidence intervals
    """
    
    def simple_forecast(data, horizon):
        """Simple forecasting method for ensemble"""
        if len(data) == 0:
            return np.zeros(horizon)
        
        recent_values = data[demand_col].values[-30:]  # Last 30 days
        if len(recent_values) == 0:
            return np.zeros(horizon)
        
        # Simple trend + seasonal pattern
        mean_demand = np.mean(recent_values)
        trend = 0
        if len(recent_values) > 7:
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        # Generate forecast
        forecast = []
        for i in range(horizon):
            base_forecast = mean_demand + trend * i
            # Add simple seasonality (weekly pattern)
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * i / 7)
            forecast.append(max(0, base_forecast * seasonal_factor))
        
        return np.array(forecast)
    
    # Generate different forecasts
    full_history_forecast = simple_forecast(historical_data, forecast_horizon)
    post_break_forecast = simple_forecast(post_break_data, forecast_horizon)
    
    # Calculate ensemble weights based on break strength and data availability
    post_break_weight = break_strength * 0.7  # Strong breaks favor post-break data
    
    # Adjust weights based on data availability
    if len(post_break_data) < 30:  # Insufficient post-break data
        post_break_weight *= 0.5
    
    full_history_weight = 1 - post_break_weight
    
    # Combine forecasts
    ensemble_forecast = (full_history_weight * full_history_forecast + 
                        post_break_weight * post_break_forecast)
    
    # Calculate confidence interval multiplier
    # Higher break strength = more uncertainty
    confidence_multiplier = 1.0 + break_strength * 0.5
    
    # Additional uncertainty if post-break period is short
    if len(post_break_data) < 60:
        confidence_multiplier += 0.3
    
    return ensemble_forecast, confidence_multiplier

def get_production_parameters(item_data, break_info=None, base_params=None):
    """
    Enhanced production parameter function that considers structural breaks.
    """
    if base_params is None:
        base_params = {'weight': 1.0, 'regularization': 0.1}
    
    if break_info is None:
        return base_params
    
    break_points, break_strength, post_break_data = break_info
    
    # Adjust parameters based on structural break context
    enhanced_params = base_params.copy()
    
    # Increase regularization for high break strength (reduce overfitting)
    enhanced_params['regularization'] *= (1 + break_strength * 0.5)
    
    # Adjust sample weights to emphasize recent data after breaks
    if break_strength > 0.3:  # Significant break detected
        enhanced_params['weight'] *= (1 + break_strength * 0.3)
    
    return enhanced_params

def convert_to_wide_format(df, item_column, variable_column, suffix='wide'):
    df['Rank'] = df.groupby(item_column).cumcount()+1
    df_wide = df.pivot(index=item_column,
                       columns='Rank',
                       values=variable_column).reset_index()
    df_wide.columns = [item_column] + [f'{variable_column} {i}' for i in range(1, len(df_wide.columns))]
    df_wide.name = f'{df.name}_{suffix}' if hasattr(df, 'name') else f'transformed_{suffix}'
    return df_wide


def compute_adi_cv(group, demand_col):
    demand = group[demand_col].values
    non_zero_demand_intervals = np.diff(np.where(demand > 0)[0]) if (demand > 0).any() else np.array([])
    ADI = (np.mean(non_zero_demand_intervals) if len(non_zero_demand_intervals) > 0 else 0) + 1
    demand_mean = np.mean(demand)
    demand_std = np.std(demand)
    CV2 = (demand_std / demand_mean)**2 if demand_mean > 0 else 0
    return pd.Series({'ADI': ADI, 'CV2': CV2})

def wide_to_long(df, date):
    date_min = df.groupby(['ITEM_NUMBER_SITE'])[date].min().reset_index()
    date_min.rename(columns={date:'Date_min'}, inplace=True)
    date_max = df.groupby(['ITEM_NUMBER_SITE'])[date].max().reset_index()
    date_max.rename(columns={date:'Date_max'}, inplace=True)
    df_date = pd.merge(date_min, date_max, on=['ITEM_NUMBER_SITE'], how='inner', sort=False)
    df_long = pd.concat(
    df_date.apply(
        lambda row: pd.DataFrame({
            'ITEM_NUMBER_SITE': [row['ITEM_NUMBER_SITE']] * len(pd.date_range(start=row['Date_min'], end=row['Date_max'])),
            date: pd.date_range(start=row['Date_min'], end=row['Date_max'])
        }),
        axis=1
    ).tolist(),
    ignore_index=True
    )
    df_long.sort_values(['ITEM_NUMBER_SITE', date], ascending=[True, True], inplace=True)
    df_long = df_long.copy()
    return df_long

def extract_period(df, adjust, date):
    period = df.copy()

    extraction_date = period.groupby('ITEM_NUMBER_SITE')[date].transform('max')
    period['Extraction Date'] = extraction_date

    period['Observation Period'] = (period['Extraction Date'] - period[date]).dt.days

    mask = (period['Observation Period'] >= 0) & (period['Observation Period'] <= n_days_period + adjust)
    period = period[mask].drop(columns=['Observation Period', 'Extraction Date'])

    return period

def compute_purchase_repurchase_cycle(demand_analytics, date):
    results = []

    grouped = demand_analytics.groupby('ITEM_NUMBER_SITE')

    for item, group in grouped:
        group = group.sort_values(date)

        non_zero_demand_dates = group[group['ACTUAL_DEMAND'] > 0][date]
        intervals = non_zero_demand_dates.diff().dt.days.dropna()

        mean_interval = intervals.mean()
        std_interval = intervals.std()
        num_non_zero = non_zero_demand_dates.shape[0]
        
        group['YEAR'] = group[date].dt.year
        group['MONTH'] = group[date].dt.month
        monthly_demand = group.groupby(['YEAR', 'MONTH'])['ACTUAL_DEMAND'].sum()
        num_months_with_demand = (monthly_demand > 0).sum()
        total_months = len(monthly_demand)

        results.append({
            'ITEM_NUMBER_SITE': item,
            'MEAN_PURCHASE_INTERVAL_DAYS': mean_interval,
            'Std Purchase Interval (days)': std_interval,
            'Non-zero Demand Days': num_non_zero,
            'Months With Demand': num_months_with_demand,
            'Total Months': total_months,
            'Monthly Demand Frequency': num_months_with_demand / total_months if total_months > 0 else 0
        })

    purchase_repurchase_cycle = pd.DataFrame(results)

    return purchase_repurchase_cycle

def categorize_purchase_cycle(mean_interval, num_non_zero, months_with_demand=None, monthly_frequency=None):
    if months_with_demand is not None and monthly_frequency is not None:
        if months_with_demand <= 2:
            return 'Rarely'
        elif months_with_demand <= 4 and monthly_frequency <= 0.4:  # ≤4 months AND ≤40% frequency
            return 'Bi-annual'
        elif monthly_frequency >= 0.6:  # ≥60% of months have demand
            if pd.notna(mean_interval):
                if mean_interval <= 7:
                    return 'Weekly'
                elif mean_interval <= 30:
                    return 'Monthly'
            return 'Monthly'  # Default for frequent demand
        elif months_with_demand >= 6:
            return 'Monthly'
        elif monthly_frequency >= 0.4:  # 40-60% frequency = use mean interval to decide
            if pd.notna(mean_interval) and mean_interval <= 60:  # ~2 months average
                return 'Monthly'
            else:
                return 'Bi-annual'
    
    if num_non_zero <= 2:
        return 'Rarely'
    elif num_non_zero <= 4:
        return 'Bi-annual'
    if pd.isna(mean_interval):
        return 'Rarely'
    if mean_interval <= 1:
        return 'Daily'
    elif mean_interval <= 7:
        return 'Weekly'
    elif mean_interval <= 30:
        return 'Monthly'
    elif mean_interval <= 182:
        return 'Bi-annual'
    elif mean_interval <= 365:
        return 'Yearly'
    else:
        return 'Rarely'

def compute_purchase_repurchase_cycle2(demand_analytics, date):
    results = []

    grouped = demand_analytics.groupby('ITEM_NUMBER_SITE')

    for item, group in grouped:
        group = group.sort_values(date)

        non_zero_demand_dates = group[group['ACTUAL_DEMAND'] > 0][date]
        intervals = non_zero_demand_dates.diff().dt.days.dropna()

        mean_interval = intervals.mean()
        std_interval = intervals.std()
        num_non_zero = non_zero_demand_dates.shape[0]

        results.append({
            'ITEM_NUMBER_SITE': item,
            'MEAN_PURCHASE_INTERVAL_DAYS': mean_interval,
            'Std Purchase Interval (days)': std_interval,
            'Non-zero Demand Days': num_non_zero
        })

    purchase_repurchase_cycle2 = pd.DataFrame(results)

    return purchase_repurchase_cycle2

def categorize_purchase_cycle2(mean_interval, num_non_zero):
    if num_non_zero <= 2:
        return 'Rarely'
    elif num_non_zero <= 4:
        return 'Bi-annual'
    if pd.isna(mean_interval):
        return 'Rarely'
    if mean_interval <= 1:
        return 'Daily'
    elif mean_interval <= 7:
        return 'Weekly'
    elif mean_interval <= 30:
        return 'Monthly'
    elif mean_interval <= 182:
        return 'Bi-annual'
    elif mean_interval <= 365:
        return 'Yearly'
    else:
        return 'Rarely'

def build_fit_weekly_LGBMRegressor(
    X_full,
    y_full,
    num_leaves=None,
    max_depth=None,
    learning_rate=None,
    n_estimators=None,
    min_child_samples=None,
    min_split_gain=None,
    lambda_l1=None,
    lambda_l2=None,
    subsample=None,
    colsample_bytree=None,
    feature_fraction=None,
    bagging_fraction=None,
    bagging_freq=None,
    max_bin=None,
    #device_type="gpu",
    gpu_platform_id=0,
    random_state=42,
    objective="poisson",
    alpha=None,
    monotone_constraints=None,
    monotone_constraints_method=None,
):
    model = lgb.LGBMRegressor(
        objective=objective,
        boosting_type="gbdt",
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        min_split_gain=min_split_gain,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        max_bin=max_bin,
        #device_type=device_type,
        gpu_platform_id=gpu_platform_id,
        random_state=random_state,
        verbosity=-1,
        alpha=alpha,
        monotone_constraints=monotone_constraints,
        monotone_constraints_method="intermediate",
    )
    
    # Safety check: Ensure no negative target values for Poisson regression
    if hasattr(y_full, 'values'):
        y_values = y_full.values
    else:
        y_values = y_full
    
    if np.any(y_values < 0):
        negative_count = np.sum(y_values < 0)
        print(f"Warning: Found {negative_count} negative target values, clipping to 0")
        if hasattr(y_full, 'values'):
            y_full = y_full.clip(lower=0)
        else:
            y_full = np.clip(y_full, 0, None)

    # Guard: if all targets are zero, LightGBM with poisson objective will fail
    if np.sum(y_values) == 0:
        print("[DEBUG] All-zero target vector detected in build_fit_weekly_LGBMRegressor. Returning zero predictor model.")
        class _ZeroPredictor:
            def fit(self, X, y=None, **kwargs):
                return self
            def predict(self, X):
                try:
                    n = X.shape[0]
                except Exception:
                    n = len(X)
                return np.zeros(n)
        return _ZeroPredictor()

    # Guard: if all targets are zero, LightGBM with poisson objective will fail
    if np.sum(y_values) == 0:
        print("[DEBUG] All-zero target vector detected in build_fit_less_than_weekly_LGBMRegressor. Returning zero predictor model.")
        class _ZeroPredictor:
            def fit(self, X, y=None, **kwargs):
                return self
            def predict(self, X):
                try:
                    n = X.shape[0]
                except Exception:
                    n = len(X)
                return np.zeros(n)
        return _ZeroPredictor()
    
    try:
        import cupy as cp
        # Test CuPy functionality before proceeding
        try:
            test_array = cp.array([1, 2, 3])
            test_result = test_array.get()  # This will fail if CUDA setup is broken
            print(f"CuPy test successful: {test_result}")
        except Exception as cuda_test_error:
            print(f"CuPy available but CUDA runtime issue: {cuda_test_error}")
            raise RuntimeError("CUDA runtime test failed")
            
        # Convert to CuPy for memory efficiency, then back to NumPy for LightGBM sklearn interface
        if hasattr(X_full, 'values'):
            X_gpu = cp.asarray(X_full.values)
        else:
            X_gpu = cp.asarray(X_full)
            
        if hasattr(y_full, 'values'):
            y_gpu = cp.asarray(y_full.values)
        else:
            y_gpu = cp.asarray(y_full)
        
        print(f"GPU zero-copy: X shape {X_gpu.shape}, y shape {y_gpu.shape}")
        
        # Convert back to NumPy for sklearn interface - LightGBM will handle GPU internally
        X_numpy = X_gpu.get()
        y_numpy = y_gpu.get()
        
        model.fit(
            X_numpy, y_numpy,
            eval_metric="mae",
            callbacks=[
            lgb.log_evaluation(period=0 if ENABLE_LGBM_PRINTS else -1)
            ]
        )
    except (ImportError, RuntimeError) as e:
        if "CUDA" in str(e):
            print(f"CUDA configuration issue: {e}")
            print("Falling back to CPU arrays - LightGBM will still use GPU via device='gpu'")
        else:
            print("CuPy not available, falling back to CPU arrays")
        model.fit(
            X_full, y_full,
            eval_metric="mae",
            callbacks=[
            lgb.log_evaluation(period=0 if ENABLE_LGBM_PRINTS else -1)
            ]
        )
    
    return model


def objective_weekly_LGBM_with_weight(trial, X_train, y_train, X_val, y_val, wgt, enhanced_regularization=0.1):
    try:
        param = {
            "objective": "poisson",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
            "bagging_seed": RANDOM_SEED,
            "device_type": "gpu",
            "gpu_platform_id": 0,
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, enhanced_regularization * 50),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, enhanced_regularization * 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "max_bin": trial.suggest_int("max_bin", 3, 100),
        }
        train_weights = np.where(y_train > 0, wgt, 1)
        val_weights = np.where(y_val > 0, wgt, 1)

        try:
            # Test CuPy functionality first with better error handling
            try:
                test_array = cp.array([1, 2, 3])
                test_result = test_array.get()
                # If we get here, CuPy is working properly
            except Exception as cuda_test_error:
                if "CUDA root directory" in str(cuda_test_error):
                    # This is the specific CUDA path issue - use fallback silently
                    raise RuntimeError("CUDA path detection failed - using CPU fallback")
                else:
                    raise cuda_test_error
            
            if hasattr(X_train, 'values'):
                X_train_gpu = cp.asarray(X_train.values)
            else:
                X_train_gpu = cp.asarray(X_train)
                
            if hasattr(y_train, 'values'):
                y_train_gpu = cp.asarray(y_train.values)
            else:
                y_train_gpu = cp.asarray(y_train)
            
            if hasattr(X_val, 'values'):
                X_val_gpu = cp.asarray(X_val.values)
            else:
                X_val_gpu = cp.asarray(X_val)
                
            if hasattr(y_val, 'values'):
                y_val_gpu = cp.asarray(y_val.values)
            else:
                y_val_gpu = cp.asarray(y_val)
            
            train_weights_gpu = cp.asarray(train_weights)
            val_weights_gpu = cp.asarray(val_weights)
            
            # Convert all arrays to NumPy for LightGBM sklearn interface
            X_train_numpy = X_train_gpu.get()
            y_train_numpy = y_train_gpu.get()
            X_val_numpy = X_val_gpu.get()
            y_val_numpy = y_val_gpu.get()
            train_weights_numpy = train_weights_gpu.get()
            val_weights_numpy = val_weights_gpu.get()
            
            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_train_numpy, y_train_numpy,
                sample_weight=train_weights_numpy,
                eval_set=[(X_val_numpy, y_val_numpy)],
                eval_sample_weight=[val_weights_numpy],
                eval_metric="mae",
                verbose=False,
                callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=0 if ENABLE_LGBM_PRINTS else -1)
                ]
            )
            preds = model.predict(X_val_numpy)
            mae = mean_absolute_error(y_val_numpy, preds, sample_weight=val_weights_numpy)
            
        except (ImportError, RuntimeError) as e:
            # Silently fall back to CPU - LightGBM will still use GPU internally
            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_train, y_train,
                sample_weight=train_weights,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[val_weights],
                eval_metric="mae",
                verbose=False,
                callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=0 if ENABLE_LGBM_PRINTS else -1)
                ]
            )
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds, sample_weight=val_weights)
            
        return mae
    except Exception as e:
        if ENABLE_OPTUNA_PRINTS:
            print(f"Weekly LGBM trial failed with error: {e}")
        return float('inf')

def build_fit_less_than_weekly_LGBMRegressor(
    X_full,
    y_full,
    num_leaves=None,
    max_depth=None,
    learning_rate=None,
    n_estimators=None,
    min_child_samples=None,
    min_split_gain=None,
    lambda_l1=None,
    lambda_l2=None,
    subsample=None,
    colsample_bytree=None,
    feature_fraction=None,
    bagging_fraction=None,
    bagging_freq=None,
    max_bin=None,
    min_data_in_leaf=None,
    device_type="gpu",
    gpu_platform_id=0,
    random_state=42,
    objective="poisson", # regression
    alpha=None,
    monotone_constraints=None,
    monotone_constraints_method=None,
    
):
    model = lgb.LGBMRegressor(
        objective=objective,
        boosting_type="gbdt",
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        min_split_gain=min_split_gain,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        max_bin=max_bin,
        min_data_in_leaf=min_data_in_leaf,
        device_type=device_type,
        gpu_platform_id=gpu_platform_id,
        random_state=random_state,
        verbosity=-1,
        alpha=alpha,
        monotone_constraints=monotone_constraints,
        monotone_constraints_method="intermediate",
    )   
    
    # Safety check: Ensure no negative target values for Poisson regression
    if hasattr(y_full, 'values'):
        y_values = y_full.values
    else:
        y_values = y_full
    
    if np.any(y_values < 0):
        negative_count = np.sum(y_values < 0)
        print(f"Warning: Found {negative_count} negative target values, clipping to 0")
        if hasattr(y_full, 'values'):
            y_full = y_full.clip(lower=0)
        else:
            y_full = np.clip(y_full, 0, None)

    # Guard: if all targets are zero, LightGBM with poisson objective will fail
    if np.sum(y_values) == 0:
        print("[DEBUG] All-zero target vector detected in build_fit_weekly_LGBMRegressor. Returning zero predictor model.")
        class _ZeroPredictor:
            def fit(self, X, y=None, **kwargs):
                return self
            def predict(self, X):
                try:
                    n = X.shape[0]
                except Exception:
                    n = len(X)
                return np.zeros(n)
        return _ZeroPredictor()

    # Guard: if all targets are zero, LightGBM with poisson objective will fail
    if np.sum(y_values) == 0:
        print("[DEBUG] All-zero target vector detected in build_fit_less_than_weekly_LGBMRegressor. Returning zero predictor model.")
        class _ZeroPredictor:
            def fit(self, X, y=None, **kwargs):
                return self
            def predict(self, X):
                try:
                    n = X.shape[0]
                except Exception:
                    n = len(X)
                return np.zeros(n)
        return _ZeroPredictor()
    
    try:
        import cupy as cp
        # Test CuPy functionality before proceeding
        try:
            test_array = cp.array([1, 2, 3])
            test_result = test_array.get()
            print(f"CuPy test successful for less-than-weekly: {test_result}")
        except Exception as cuda_test_error:
            print(f"CuPy available but CUDA runtime issue: {cuda_test_error}")
            raise RuntimeError("CUDA runtime test failed")
            
        if hasattr(X_full, 'values'):  # pandas DataFrame
            X_gpu = cp.asarray(X_full.values)
        else:
            X_gpu = cp.asarray(X_full)
            
        if hasattr(y_full, 'values'):
            y_gpu = cp.asarray(y_full.values)
        else:
            y_gpu = cp.asarray(y_full)
        
        print(f"GPU zero-copy (less-than-weekly): X shape {X_gpu.shape}, y shape {y_gpu.shape}")
        
        # Convert back to NumPy for sklearn interface - LightGBM will handle GPU internally
        X_numpy = X_gpu.get()
        y_numpy = y_gpu.get()
        
        model.fit(
            X_numpy, y_numpy,
            eval_metric="mae",
            callbacks=[
            lgb.log_evaluation(period=0 if ENABLE_LGBM_PRINTS else -1)
            ]
        )
    except (ImportError, RuntimeError) as e:
        if "CUDA" in str(e):
            print(f"CUDA configuration issue: {e}")
            print("Falling back to CPU arrays - LightGBM will still use GPU via device='gpu'")
        else:
            print("CuPy not available, falling back to CPU arrays")
        model.fit(
            X_full, y_full,
            eval_metric="mae",
            callbacks=[
            lgb.log_evaluation(period=0 if ENABLE_LGBM_PRINTS else -1)
            ]
        )
        
    return model


def objective_less_than_weekly_LGBM_with_weight(trial, X_train, y_train, X_val, y_val, wgt, enhanced_regularization=0.1):
    try:
        param = {
            "objective": "poisson",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": RANDOM_SEED,
            "feature_fraction_seed": RANDOM_SEED,
            "bagging_seed": RANDOM_SEED,
            "device_type": "gpu",
            "gpu_platform_id": 0,
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, enhanced_regularization * 50),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, enhanced_regularization * 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 20),
            "max_bin": trial.suggest_int("max_bin", 3, 100),
        }
        train_weights = np.where(y_train > 0, wgt, 1)
        val_weights = np.where(y_val > 0, wgt, 1)
        
        try:
            # Test CuPy functionality first with better error handling
            try:
                test_array = cp.array([1, 2, 3])
                test_result = test_array.get()
                # If we get here, CuPy is working properly
            except Exception as cuda_test_error:
                if "CUDA root directory" in str(cuda_test_error):
                    # This is the specific CUDA path issue - use fallback silently
                    raise RuntimeError("CUDA path detection failed - using CPU fallback")
                else:
                    raise cuda_test_error
            
            if hasattr(X_train, 'values'):
                X_train_gpu = cp.asarray(X_train.values)
            else:
                X_train_gpu = cp.asarray(X_train)
                
            if hasattr(y_train, 'values'):
                y_train_gpu = cp.asarray(y_train.values)
            else:
                y_train_gpu = cp.asarray(y_train)
            
            if hasattr(X_val, 'values'):
                X_val_gpu = cp.asarray(X_val.values)
            else:
                X_val_gpu = cp.asarray(X_val)
                
            if hasattr(y_val, 'values'):
                y_val_gpu = cp.asarray(y_val.values)
            else:
                y_val_gpu = cp.asarray(y_val)
            
            train_weights_gpu = cp.asarray(train_weights)
            val_weights_gpu = cp.asarray(val_weights)
            
            X_train_numpy = X_train_gpu.get()
            y_train_numpy = y_train_gpu.get()
            X_val_numpy = X_val_gpu.get()
            y_val_numpy = y_val_gpu.get()
            train_weights_numpy = train_weights_gpu.get()
            val_weights_numpy = val_weights_gpu.get()
            
            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_train_numpy, y_train_numpy, 
                sample_weight=train_weights_numpy,
                eval_set=[(X_val_numpy, y_val_numpy)],
                eval_sample_weight=[val_weights_numpy],
                eval_metric="mae",
                verbose=False,
                callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=100 if ENABLE_LGBM_PRINTS else -1)
                ]
            )
            preds = model.predict(X_val_numpy)
            mae = mean_absolute_error(y_val_numpy, preds, sample_weight=val_weights_numpy)
            
        except (ImportError, RuntimeError) as e:
            # Silently fall back to CPU - LightGBM will still use GPU internally
            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_train, y_train,
                sample_weight=train_weights,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[val_weights],
                eval_metric="mae",
                verbose=False,
                callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=100 if ENABLE_LGBM_PRINTS else -1)
                ]
            )
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds, sample_weight=val_weights)
            
        return mae

    except Exception as e:
        if ENABLE_OPTUNA_PRINTS:
            print(f"Less-than-weekly LGBM trial failed with error: {e}")
        return float('inf')

def print_callback(study, trial):
    if ENABLE_OPTUNA_PRINTS:
        print(f"Current value: {trial.value}, Current params: {trial.params}")
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def get_previous_month_length(df):
    df_sorted = df.sort_values(['YEAR', 'MONTH'])
    unique_months = df_sorted[['YEAR', 'MONTH']].drop_duplicates().values
    if len(unique_months) < 2:
        return 0  # Not enough data for previous month
    prev_year, prev_month = unique_months[-2]
    prev_month_rows = df[(df['YEAR'] == prev_year) & (df['MONTH'] == prev_month)]
    return len(prev_month_rows)


def simulate_aggregate_intervals_norm(forecast_data, value_col='Forecast', inf_col='Forecast Inf', sup_col='Forecast Sup', n_sim=10000, group_cols=None):
    if group_cols is not None:
        results = []
        for group_vals, group_df in forecast_data.groupby(group_cols):
            agg = simulate_aggregate_intervals(group_df, value_col, inf_col, sup_col, n_sim, group_cols=None)
            for i, col in enumerate(group_cols):
                agg[col] = group_vals[i] if isinstance(group_vals, tuple) else group_vals
            results.append(agg)
        return pd.concat(results, ignore_index=True)
    
    means = forecast_data[value_col].values
    lows = forecast_data[inf_col].values
    highs = forecast_data[sup_col].values

    stds = (highs - lows) / (2 * 1.645)

    rng = np.random.default_rng(RANDOM_SEED)
    daily_samples = rng.normal(loc=means, scale=stds, size=(n_sim, len(means)))
    daily_samples = np.clip(daily_samples, a_min=0, a_max=None)  # Demand can't be negative

    agg_samples = daily_samples.sum(axis=1)

    agg_inf = np.percentile(agg_samples, 5)
    agg_med = np.percentile(agg_samples, 50)
    agg_sup = np.percentile(agg_samples, 95)

    return pd.DataFrame({
        'Aggregate Forecast': [agg_med],
        'FORECASTED_DEMAND_INF': [agg_inf],
        'FORECASTED_DEMAND_SUP': [agg_sup]
    })


def simulate_aggregate_intervals(
    forecast_data,
    value_col='ACTUAL_DEMAND', 
    inf_col='FORECASTED_DEMAND_INF', 
    sup_col='FORECASTED_DEMAND_SUP',             
    n_sim=10000, 
    group_cols=None, 
    dist='poisson'
):
    if group_cols is not None:
        results = []
        for group_vals, group_df in forecast_data.groupby(group_cols):
            agg = simulate_aggregate_intervals(group_df, value_col, inf_col, sup_col, n_sim, group_cols=None, dist=dist)
            for i, col in enumerate(group_cols):
                agg[col] = group_vals[i] if isinstance(group_vals, tuple) else group_vals
            results.append(agg)
        return pd.concat(results, ignore_index=True)

    means = forecast_data[value_col].values
    lows = forecast_data[inf_col].values
    highs = forecast_data[sup_col].values
    n_days = len(means)
    rng = np.random.default_rng(RANDOM_SEED)

    if dist == 'poisson':
        daily_samples = rng.poisson(lam=means, size=(n_sim, n_days))
    elif dist == 'nbinom':
        stds = (highs - lows) / (2 * 1.645)
        var = stds**2
        var = np.maximum(var, means)
        daily_samples = np.zeros((n_sim, n_days))
        for i in range(n_days):
            mu = means[i]
            v = var[i]
            if pd.isna(mu) or pd.isna(v) or mu <= 0:
                daily_samples[:, i] = 0
                continue
            if v > mu:
                r = (mu**2) / (v - mu + 1e-6)
                p = r / (r + mu)
                if r > 0 and 0 < p < 1 and np.isfinite(r) and np.isfinite(p):
                    daily_samples[:, i] = nbinom.rvs(r, p, size=n_sim)
                else:
                    daily_samples[:, i] = rng.poisson(lam=mu, size=n_sim)
            else:
                daily_samples[:, i] = rng.poisson(lam=mu, size=n_sim)
    else:
        raise ValueError("dist must be 'poisson' or 'nbinom'")

    agg_samples = daily_samples.sum(axis=1)
    agg_inf = np.percentile(agg_samples, 5)
    agg_med = np.percentile(agg_samples, 50)
    agg_sup = np.percentile(agg_samples, 95)

    return pd.DataFrame({
        'ACTUAL_DEMAND': [agg_med],
        'FORECASTED_DEMAND_INF': [agg_inf],
        'FORECASTED_DEMAND_SUP': [agg_sup]
    })

import re

def _force_numpy_dtypes(df):
    for col in df.select_dtypes(include=["Int8","Int16","Int32","Int64"]).columns:
        df[col] = df[col].astype("int64")
    for col in df.select_dtypes(include=["Float32","Float64"]).columns:
        df[col] = df[col].astype("float64")
    for col in df.select_dtypes(include=["boolean"]).columns:
        df[col] = df[col].astype("bool")
    for col in df.select_dtypes(include=["string"]).columns:
        df[col] = df[col].astype("object")
    return df

def clean_feature_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', str(col)) for col in df.columns
    ]
    return df

def add_rolling_features(
    df, 
    demand_col='ACTUAL_DEMAND', 
    windows=None, 
    group_col='ITEM_NUMBER_SITE'
):

    windows = windows or []
    feature_cols = []
    for window in windows:
        mean_col = f'{demand_col}_rollmean_{window}'
        std_col = f'{demand_col}_rollstd_{window}'
        skew_col = f'{demand_col}_rollskew_{window}'
        kurt_col = f'{demand_col}_rollkurt_{window}'
        zero_count_col = f'{demand_col}_rollzerocount_{window}'
        slope_col = f'{demand_col}_rollslope_{window}'
        entropy_col = f'{demand_col}_rollentropy_{window}'

        shifted = df.groupby(group_col)[demand_col].shift(1)

        df[mean_col] = shifted.rolling(window).mean().reset_index(level=0, drop=True)
        df[std_col] = shifted.rolling(window).std().reset_index(level=0, drop=True)
        df[skew_col] = shifted.rolling(window).apply(lambda x: skew(x, nan_policy='omit'), raw=True).reset_index(level=0, drop=True)
        df[kurt_col] = shifted.rolling(window).apply(lambda x: kurtosis(x, nan_policy='omit'), raw=True).reset_index(level=0, drop=True)
        df[zero_count_col] = shifted.rolling(window).apply(lambda x: np.sum(x == 0), raw=True).reset_index(level=0, drop=True)
        def rolling_slope(x):
            idx = np.arange(len(x))
            mask = ~np.isnan(x)
            if mask.sum() < 2:
                return np.nan
            return np.polyfit(idx[mask], x[mask], 1)[0]
        df[slope_col] = shifted.rolling(window).apply(rolling_slope, raw=True).reset_index(level=0, drop=True)
        def rolling_entropy(x):
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return np.nan
            value, counts = np.unique(x, return_counts=True)
            probs = counts / counts.sum()
            return entropy(probs, base=2)
        df[entropy_col] = shifted.rolling(window).apply(rolling_entropy, raw=False).reset_index(level=0, drop=True)

        feature_cols.extend([
            mean_col, std_col, skew_col, kurt_col,
            zero_count_col, slope_col, entropy_col
        ])
    return df[['PERFORM_DATE'] + feature_cols]    

def exclude_anomalous_last_year(df, group_col='ITEM_NUMBER_SITE', year_col='YEAR', days_col='EffectiveNbDays', threshold=0.3):
    df = df.copy()
    df = df.sort_values([group_col, year_col])
    last_years = df.groupby(group_col)[year_col].max().reset_index()
    df = df.merge(last_years, on=[group_col, year_col], how='left', indicator='is_last')
    df['is_last'] = df['is_last'] == 'both'
    
    def filter_group(group, include_group=None):
        if group.shape[0] < 2:
            return group  # Not enough years to compare
        last_row = group[group['is_last']]
        prev_rows = group[~group['is_last']]
        if last_row.empty or prev_rows.empty:
            return group
        median_prev = prev_rows[days_col].median()
        if last_row[days_col].iloc[0] < threshold * median_prev:
            return prev_rows
        return group

    filtered = df.groupby(group_col, group_keys=False).apply(filter_group, include_group=False)
    filtered = filtered.drop(columns=['is_last'])
    return filtered
 
def create_backtest_splits_monthly(demand_X, demand_y, idx, simulation_period_days, nb_months, simulation_period_days_prev):
    additional_months = int(idx)
    simulation_period_days = int(simulation_period_days)
    simulation_period_days_prev = int(simulation_period_days_prev)
    
    if nb_months > 1:
        backtest_val_size = 2 * int(nb_months) * simulation_period_days + 30
        backtest_train_size = 4 * 30
    else:
        backtest_val_size = 2 * max(simulation_period_days, simulation_period_days_prev)
        backtest_train_size = 3 * 30
    
    backtest_train_size += (additional_months * 30)  # Add 1 month per iteration
    
    # Extract backtesting data from HISTORICAL data only
    historical_end_idx = -(simulation_period_days + idx * simulation_period_days)
    
    total_backtest_size = backtest_train_size + backtest_val_size
    
    train_start = historical_end_idx - total_backtest_size
    train_end = historical_end_idx - backtest_val_size
    val_start = historical_end_idx - backtest_val_size  
    val_end = historical_end_idx
        
    if train_start < -len(demand_X):
        print(f"WARNING: Not enough historical data! Need {total_backtest_size} rows, have {len(demand_X)} available")
        # For cases with insufficient data, use maximum available
        available_data = len(demand_X) + historical_end_idx  # Data available before predictions
        if available_data < 90:  # Less than 3 months
            print("Insufficient data for reliable backtesting - skipping")
            return None, None, None, None
        
        backtest_train_size = int(available_data * 0.7)
        backtest_val_size = available_data - backtest_train_size
        
        train_start = historical_end_idx - backtest_train_size - backtest_val_size
        train_end = historical_end_idx - backtest_val_size
        val_start = historical_end_idx - backtest_val_size
        val_end = historical_end_idx
    
    X_backtest_val = demand_X.iloc[val_start:val_end].copy()
    X_backtest_train = demand_X.iloc[train_start:train_end].copy()
    
    y_backtest_val = demand_y.iloc[val_start:val_end].copy()
    y_backtest_train = demand_y.iloc[train_start:train_end].copy()
    
    return X_backtest_train, X_backtest_val, y_backtest_train, y_backtest_val

def create_backtest_splits_bi_annual(demand_X, demand_y, idx, simulation_period_days, nb_months, simulation_period_days_prev):
    
    # Handle the case where demand_X (includes future) is longer than demand_y (historical only)
    # We need to use demand_y length for calculating historical indices
    historical_data_length = len(demand_y)
    
    additional_months = int(idx)
    simulation_period_days = int(simulation_period_days)
    simulation_period_days_prev = int(simulation_period_days_prev)
    
    backtest_val_size = 12 * simulation_period_days
    backtest_train_size = 18 * simulation_period_days
    
    backtest_train_size += (additional_months * 30)  # Add 1 month per iteration
    
    # Extract backtesting data from HISTORICAL data only
    historical_end_idx = -(simulation_period_days + idx * simulation_period_days)
    
    total_backtest_size = backtest_train_size + backtest_val_size
    
    train_start = historical_end_idx - total_backtest_size
    train_end = historical_end_idx - backtest_val_size
    val_start = historical_end_idx - backtest_val_size  
    val_end = historical_end_idx
        
    if train_start < -len(demand_X):
        print(f"WARNING: Not enough historical data! Need {total_backtest_size} rows, have {len(demand_X)} available")
        # For cases with insufficient data, use maximum available
        available_data = len(demand_X) + historical_end_idx  # Data available before predictions
        if available_data < 90:  # Less than 3 months
            print("Insufficient data for reliable backtesting - skipping")
            return None, None, None, None
        
        backtest_train_size = int(available_data * 0.7)
        backtest_val_size = available_data - backtest_train_size
        
        train_start = historical_end_idx - backtest_train_size - backtest_val_size
        train_end = historical_end_idx - backtest_val_size
        val_start = historical_end_idx - backtest_val_size
        val_end = historical_end_idx
    
    # Handle different dataset lengths: demand_X includes future, demand_y is historical only
    # Calculate offset for demand_X indexing
    length_diff = len(demand_X) - len(demand_y)
    
    # For demand_X (includes future rows), adjust indices to account for extra rows
    X_val_start = val_start - length_diff if length_diff > 0 else val_start
    X_val_end = val_end - length_diff if length_diff > 0 else val_end
    X_train_start = train_start - length_diff if length_diff > 0 else train_start
    X_train_end = train_end - length_diff if length_diff > 0 else train_end
    
    # For demand_y (historical only), use original indices
    X_backtest_val = demand_X.iloc[X_val_start:X_val_end].copy()
    X_backtest_train = demand_X.iloc[X_train_start:X_train_end].copy()
    
    # Extract just the demand column values (not the DataFrame with PERFORM_DATE)
    # demand_y is a DataFrame with ['PERFORM_DATE', demand_col], we want just demand_col
    demand_col_name = demand_y.columns[1]  # Should be the demand column (second column)
    y_backtest_val = demand_y.iloc[val_start:val_end][demand_col_name].copy()
    y_backtest_train = demand_y.iloc[train_start:train_end][demand_col_name].copy()
    
    
    return X_backtest_train, X_backtest_val, y_backtest_train, y_backtest_val

def distribute_monthly_demand(forecast, target_monthly_demand, demand_days, month, demand_col, diff):
    """
    Distribute monthly demand across days using business logic for realistic patterns.
    
    Args:
        forecast: DataFrame with forecast data
        target_monthly_demand: Total monthly demand to distribute
        demand_days: Business logic number of days (from demand volume)
        month: Month to distribute within
        demand_col: Column name for actual demand
        diff: Difference to add/subtract
        
    Returns:
        None (modifies forecast DataFrame in place)
    """
    pos_indices = forecast[(forecast['MONTH'] == month) & (forecast[demand_col] > 0)].index
    n_pos = len(pos_indices)
    
    # Use business logic demand_days for realistic distribution
    optimal_days = max(min(demand_days, 10), 1)  # Cap at 10, min 1
    possible_days = forecast[(forecast['MONTH'] == month)].index
    
    if diff > 0:  # Need to increase
        if n_pos >= optimal_days:
            # Use existing positive days if we have enough
            selected_indices = pos_indices[:optimal_days]
            add_per_day = int(np.ceil(diff / len(selected_indices)))
            forecast.loc[selected_indices, demand_col] += add_per_day
            forecast.loc[selected_indices, 'FORECASTED_DEMAND_INF'] += add_per_day
            forecast.loc[selected_indices, 'FORECASTED_DEMAND_SUP'] += add_per_day
            print(f"Added {add_per_day} units per day to {len(selected_indices)} existing positive days")
        elif len(possible_days) >= 21:
            # Create new demand days using business logic
            if n_pos > 0:
                # Keep existing positive days and add more
                remaining_days_needed = optimal_days - n_pos
                available_days = [idx for idx in possible_days[5:21] if idx not in pos_indices]
                if len(available_days) >= remaining_days_needed:
                    # Use seeded random for deterministic behavior
                    random.seed(RANDOM_SEED)

                    additional_indices = random.sample(available_days, remaining_days_needed)
                    selected_indices = list(pos_indices) + additional_indices
                else:
                    selected_indices = list(pos_indices) + available_days
            else:
                # No existing positive days, create all new ones
                random.seed(RANDOM_SEED)

                selected_indices = random.sample(list(possible_days)[5:21], min(optimal_days, len(possible_days)-5))
            
            # Distribute demand across selected days
            add_per_day = int(np.ceil(target_monthly_demand / len(selected_indices)))
            for idx in selected_indices:
                forecast.at[idx, demand_col] = add_per_day
                forecast.at[idx, 'FORECASTED_DEMAND_INF'] = int(add_per_day * 0.95)
                forecast.at[idx, 'FORECASTED_DEMAND_SUP'] = int(add_per_day * 1.15)
            print(f"Distributed {target_monthly_demand:.0f} units across {len(selected_indices)} days (business logic: {demand_days} days)")
        else:
            # Fallback: use whatever positive days we have or create minimal distribution
            selected_indices = pos_indices if n_pos > 0 else possible_days[:1]
            if len(selected_indices) > 0:
                add_per_day = int(np.ceil(target_monthly_demand / len(selected_indices)))
                for idx in selected_indices:
                    forecast.at[idx, demand_col] = add_per_day
                    forecast.at[idx, 'FORECASTED_DEMAND_INF'] = int(add_per_day * 0.95)
                    forecast.at[idx, 'FORECASTED_DEMAND_SUP'] = int(add_per_day * 1.15)
                print(f"Fallback distribution: {target_monthly_demand:.0f} units across {len(selected_indices)} days")


ts_format = "%Y-%m-%d"  #" %H:%M:%S"
ts_format_2 = "%Y-%m-%d %H:%M:%S"
ts_format_2b = "%m/%d/%y"
ts_format_3 = "%m/%d/%y %H:%M:%S"
division = 'Trucks' #'Trucks'  'UK'



# not referenced later
#demand_reamonn = pd.read_csv(fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\test\demand.csv', encoding='UTF-8', sep=",", decimal='.', doublequote=False, engine='python')
#demand_reamonn['PERFORM_DATE'] = pd.to_datetime(demand_reamonn['PERFORM_DATE'], format=ts_format)
#date_max = demand_reamonn.groupby(["ITEM_NUMBER_SITE"])['PERFORM_DATE'].max().reset_index()


"""
demand_reamonn = pd.read_csv(fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\test\demand.csv', encoding='UTF-8', sep=",", decimal='.', doublequote=False, engine='python')
demand_reamonn['Perform Date'] = pd.to_datetime(demand_reamonn['Perform Date'], format=ts_format)
date_max = demand_reamonn.groupby(["Item Number Site"])['Perform Date'].max().reset_index()

demand = pd.read_csv(fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\demand_sept_2025.csv', encoding='UTF-8', sep="|", decimal='.', doublequote=True, engine='python')
demand['Perform Date'] = pd.to_datetime(demand['Perform Date'], format=ts_format)
date_max_fr = demand.groupby(["Item Number Site"])['Perform Date'].max().reset_index()

demand.loc[demand['Item Number Site'] == '13AXC0204_400_HENDUK', 'Item Number Site'] = '13AXC0424_400_HENDUK'
demand.loc[demand['Item Number Site'] == '13AXC0222_400_HENDUK', 'Item Number Site'] = '13AXC0420_400_HENDUK'
demand = demand.sort_values('Actual Demand')  # Sort so min is first
demand = demand.drop_duplicates(subset=['Item Number Site', 'Perform Date'], keep='first')
demand.sort_values(['Item Number Site', 'Perform Date'], ascending=[True,True], inplace=True)
zz = demand[demand['Item Number Site']=='41CXZ0018P_400_HENDUK']
demand_cnt = demand.groupby('Item Number Site')['Perform Date'].max().reset_index()
file_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\Hendrickson\{division}\Datas_in\tr_hist.csv'
timestamp = os.path.getmtime(file_path)
demand_cnt['date_extract'] = pd.to_datetime(datetime.fromtimestamp(timestamp))
demand_cnt['eligible'] = (demand_cnt['date_extract'] - demand_cnt['Perform Date']).dt.days

demand_cnt = demand_cnt[demand_cnt['eligible'] <=120].copy()
#demand = pd.merge(demand, demand_cnt[['Item Number Site']], on=['Item Number Site'], how='inner', sort=False)

n_days_period = 365*2

demand['YEAR'] = demand['Perform Date'].dt.year
demand.drop(columns={'YEAR'}, inplace=True)
demand_analytics = extract_period(demand, 1, 'Perform Date')
demand_period_long = wide_to_long(demand_analytics, 'Perform Date')
demand_period_long.sort_values(['Item Number Site', 'Perform Date'], ascending=[True, True], inplace=True)
demand_period_long = pd.merge(demand_period_long, demand_analytics, on=['Item Number Site', 'Perform Date'], how='left', sort=False)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].fillna(0)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].round(0)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].astype(int)

purchase_repurchase_cycle = compute_purchase_repurchase_cycle(demand_period_long, 'Perform Date')
purchase_repurchase_cycle['Purchase Cycle Category'] = purchase_repurchase_cycle.apply(
    lambda row: categorize_purchase_cycle(
        row['Mean Purchase Interval (days)'], 
        row['Non-zero Demand Days'],
        row.get('Months With Demand'),
        row.get('Monthly Demand Frequency')
    ),
    axis=1
)
n_days_period = 365*1
demand_analytics = extract_period(demand, 1, 'Perform Date')
demand_period_long = wide_to_long(demand_analytics, 'Perform Date')
demand_period_long.sort_values(['Item Number Site', 'Perform Date'], ascending=[True, True], inplace=True)
demand_period_long = pd.merge(demand_period_long, demand_analytics, on=['Item Number Site', 'Perform Date'], how='left', sort=False)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].fillna(0)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].round(0)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].astype(int)
purchase_repurchase_cycle2 = compute_purchase_repurchase_cycle2(demand_period_long, 'Perform Date')
purchase_repurchase_cycle2['Purchase Cycle Category2'] = purchase_repurchase_cycle2.apply(
    lambda row: categorize_purchase_cycle2(row['Mean Purchase Interval (days)'], row['Non-zero Demand Days']),
    axis=1
)
purchase_repurchase_cycle = pd.merge(purchase_repurchase_cycle, purchase_repurchase_cycle2[['Item Number Site','Purchase Cycle Category2']], on=['Item Number Site'], how='inner', sort=False)
purchase_repurchase_cycle['Purchase Cycle Category'] = np.where(purchase_repurchase_cycle['Purchase Cycle Category']!=purchase_repurchase_cycle['Purchase Cycle Category2'],
                                                                purchase_repurchase_cycle['Purchase Cycle Category2'],
                                                                purchase_repurchase_cycle['Purchase Cycle Category'])
purchase_repurchase_cycle.drop(columns={'Purchase Cycle Category2'}, inplace=True)

n_days_period = 365*4

demand_analytics = extract_period(demand, 1, 'Perform Date')
demand_period_long = wide_to_long(demand_analytics, 'Perform Date')
demand_period_long.sort_values(['Item Number Site', 'Perform Date'], ascending=[True, True], inplace=True)
demand_period_long = pd.merge(demand_period_long, demand_analytics, on=['Item Number Site', 'Perform Date'], how='left', sort=False)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].fillna(0)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].round(0)
demand_period_long['Actual Demand'] = demand_period_long['Actual Demand'].astype(int)
demand_period_long.drop(columns={'Item Number', 'Site', 'Domain'}, inplace=True)

demand_col = 'Actual Demand'


monthly_tot_all_items = demand_period_long.copy()
monthly_tot_all_items['YEAR'] = monthly_tot_all_items['Perform Date'].dt.year
monthly_tot_all_items['MONTH'] = monthly_tot_all_items['Perform Date'].dt.month
monthly_tot_all_items['DaysInMonth'] = monthly_tot_all_items['Perform Date'].dt.days_in_month
monthly_tot_all_items['EffectiveNbDays'] = 1
monthly_tot_all_items['EffectiveDemand'] = np.where(monthly_tot_all_items[demand_col] > 0, 1, 0)
monthly_tot_all_items = monthly_tot_all_items.groupby(['Item Number Site','YEAR','MONTH']).agg({'DaysInMonth':'max',                                                                 
                                                          'EffectiveNbDays': 'sum',
                                                          'EffectiveDemand': 'sum',
                                                          demand_col: 'sum'}).reset_index()

def is_incomplete(row):
    if row['MONTH'] == 2:
        return row['EffectiveNbDays'] < 28
    else:
        return row['EffectiveNbDays'] < 30

monthly_tot_all_items = monthly_tot_all_items.sort_values(['Item Number Site', 'YEAR', 'MONTH'])
mask = monthly_tot_all_items.groupby('Item Number Site').tail(1).apply(is_incomplete, axis=1)
last_indices = monthly_tot_all_items.groupby('Item Number Site').tail(1).index
monthly_tot_all_items = monthly_tot_all_items.drop(index=last_indices[mask])

years_per_item = monthly_tot_all_items.groupby('Item Number Site')['YEAR'].nunique().reset_index()

years_per_item.rename(columns={'YEAR': 'NumYears'}, inplace=True)
years_per_item = years_per_item[years_per_item['NumYears']>3].copy()
purchase_repurchase_cycle = pd.merge(purchase_repurchase_cycle, years_per_item[['Item Number Site']], on=['Item Number Site'], how='inner', sort = False)

purchase_repurchase_cycle2 = pd.merge(purchase_repurchase_cycle2, purchase_repurchase_cycle[['Item Number Site','Mean Purchase Interval (days)']], on=['Item Number Site'], how='left', sort=False)
purchase_repurchase_cycle2 = purchase_repurchase_cycle2[pd.isna(purchase_repurchase_cycle2['Mean Purchase Interval (days)_y'])].copy()
purchase_repurchase_cycle2.rename(columns={'Purchase Cycle Category2': 'Purchase Cycle Category'}, inplace=True)

years_per_item = monthly_tot_all_items.groupby('Item Number Site')['YEAR'].nunique().reset_index()
years_per_item.rename(columns={'YEAR': 'NumYears'}, inplace=True)
years_per_item = years_per_item[years_per_item['NumYears']>1].copy()
purchase_repurchase_cycle2 = pd.merge(purchase_repurchase_cycle2, years_per_item[['Item Number Site']], on=['Item Number Site'], how='inner', sort = False)

item_selection_bi_annual = purchase_repurchase_cycle2[purchase_repurchase_cycle2['Purchase Cycle Category']=='Bi-annual']
item_selection = purchase_repurchase_cycle[purchase_repurchase_cycle['Purchase Cycle Category'].isin(['Monthly', 'Weekly'])]
zz = demand[demand['Item Number Site']=='000006-000C_102_HENDTKUS']

demo = pd.merge(demand_period_long, item_selection[['Item Number Site']], on=['Item Number Site'], how='inner', sort=False)
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\demo.csv'
demo.to_csv(export_path, index=None, doublequote=False, header=True, sep=",", encoding='UTF-8')
"""
def reduce_mem_usage(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        # Check if it fits in int32 to save even more
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')
    return df
"""
zz1 = purchase_repurchase_cycle[purchase_repurchase_cycle['Item Number Site']=='047691-000_101_HENDTKUS']
zz2 = purchase_repurchase_cycle2[purchase_repurchase_cycle2['Item Number Site']=='047691-000_101_HENDTKUS']

item_selection = item_selection[item_selection['Item Number Site'].isin(['HF100259_400_HENDUK','41CXZ0018P_400_HENDUK','62001-640_400_HENDUK','15CXC0031P_400_HENDUK','10ARC0018_400_HENDUK'])]
item_selection_bi_annual = item_selection_bi_annual[item_selection_bi_annual['Item Number Site'].isin(['HS507425_400_HENDUK','30AXC0404_400_HENDUK','HS508656_400_HENDUK','11CLC0410_400_HENDUK','Y787110/LH_400_HENDUK'])]

item_selection = purchase_repurchase_cycle[purchase_repurchase_cycle['Item Number Site'].isin(['047691-000_101_HENDTKUS'])]
"""

#item_selection = item_selection_w.head(8).copy()

horizon = 3

start_time = time.time()

final_forecast = pd.DataFrame()
final_intervals = pd.DataFrame()
all_forecasts = []
all_summaries = []
all_global_interval = []
all_last_years = []
all_best_params = []
final_params = pd.DataFrame()
all_best_feats = []
final_feats = pd.DataFrame()
demand_col = 'ACTUAL_DEMAND'
item_cnt = 0

def forecast_partition(df: pd.DataFrame) -> pd.DataFrame:
    """MMT worker: run weekly demand forecast for one partition (item), upload results to stage."""

    import pandas as pd
    import numpy as np
    import gc

    import lightgbm as lgb

    from tsfresh import extract_features, extract_relevant_features, select_features
    from tsfresh.utilities.dataframe_functions import impute, roll_time_series
    
    from darts import TimeSeries, concatenate
    from darts.dataprocessing.transformers import Scaler, Mapper, InvertibleMapper
    from darts.models import KalmanForecaster, LinearRegressionModel
    from darts.models.filtering.kalman_filter import KalmanFilter
    from darts.utils.timeseries_generation import datetime_attribute_timeseries
    from darts.metrics import smape, mape, rmse, r2_score, mase, mae
    from sklearn.metrics import mean_absolute_error
    from sklego.preprocessing import RepeatingBasisFunction

    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    from statsmodels.tools.sm_exceptions import ValueWarning
    
    warnings.filterwarnings('ignore', category=ValueWarning)
    warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
    warnings.filterwarnings('ignore', message="No supported index is available")
    warnings.filterwarnings('ignore', message="A date index has been provided, but it has no associated frequency")
    
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.stattools import acf, adfuller
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels.tsa.holtwinters')
    from darts import TimeSeries
    from darts.metrics import mae
    import scipy.stats as st
    from sklearn.preprocessing import StandardScaler
    import concurrent.futures
    
    from mapie.regression import SplitConformalRegressor
    from mapie.utils import train_conformalize_test_split

    #import cupy as cp # not supported on my mac
    from datetime import datetime, timedelta
    import math
    from scipy import stats
    from scipy.stats import skew, kurtosis, entropy, shapiro, kstest
    import random
    from scipy.stats import spearmanr, poisson, nbinom # gaussian_kde, norm, expon, lognorm, exponweib, weibull_min, gamma
    import sys
    import time
    import os
    import calendar
    import gc
    import ast
    from weekly_features_for_daily_forecasts import features_w_14_all, features_prev_M_all, features_prev_6M_all
    
        
            
    
    import optuna
    from optuna.pruners import HyperbandPruner
    import logging
    
    ENABLE_OPTUNA_PRINTS = False   # Set to True to see optimization progress
    ENABLE_LGBM_PRINTS = False    # Set to True to see LightGBM training progress
    
    if not ENABLE_OPTUNA_PRINTS:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    
    from optuna.samplers import TPESampler
    from datetime import datetime

    if df.empty:
        return None

    input_df.columns = [c.upper() for c in input_df.columns]
    input_df['PERFORM_DATE'] = pd.to_datetime(input_df['PERFORM_DATE'], format=ts_format, utc=True).dt.tz_localize(None)
    for col in input_df.select_dtypes(include=["object"]).columns:
        if col == "PERFORM_DATE":
            continue
        try:
            input_df[col] = pd.to_numeric(input_df[col], errors="ignore")
        except (ValueError, TypeError):
            pass
    numeric_cols = input_df.select_dtypes(include=["number"]).columns
    input_df[numeric_cols] = input_df[numeric_cols].astype("float64")
    input_df = input_df.sort_values("PERFORM_DATE").reset_index(drop=True)
    actual_category = input_df['PURCHASE_CYCLE_CATEGORY'].iloc[0]
    item_id = input_df['ITEM_NUMBER_SITE'].iloc[0]
    print(f"Processing item {item_id} with category: {actual_category}")
    demand_item_fct = input_df.copy()
        
    # Initialize variables to avoid NameError issues
    monthly_forecast_max = 0  # Will be properly set in monthly processing, default to 0 for weekly items
    forecast_data = None

    demand_item_fct.reset_index(drop=True, inplace=True)
    demand_item_fct['YEAR'] = demand_item_fct['PERFORM_DATE'].dt.year
    demand_item_fct['MONTH'] = demand_item_fct['PERFORM_DATE'].dt.month
    demand_item_fct['DaysInMonth'] = demand_item_fct['PERFORM_DATE'].dt.days_in_month
    demand_item_fct['EffectiveNbDays'] = 1
    demand_item_fct['EffectiveDemand'] = np.where(demand_item_fct[demand_col] > 0, 1, 0)
    monthly_tot = demand_item_fct.groupby(['YEAR','MONTH']).agg({'DaysInMonth':'max',                                                                 
                                                            'EffectiveNbDays': 'sum',
                                                            'EffectiveDemand': 'sum',
                                                            demand_col: 'sum'}).reset_index()
    if (
    (monthly_tot.iloc[-1]['EffectiveNbDays'] < 28 and monthly_tot.iloc[-1]['MONTH'] == 2) or
    (monthly_tot.iloc[-1]['EffectiveNbDays'] < 30 and monthly_tot.iloc[-1]['MONTH'] != 2)
    ):
        monthly_tot = monthly_tot.iloc[:-1]
        
    demand_item_fct = pd.merge(demand_item_fct, monthly_tot[['YEAR','MONTH']], on=['YEAR','MONTH'], how='inner', sort=False)
    
    if actual_category in ["Daily", "Weekly"]:
        if (
        (monthly_tot.iloc[0]['EffectiveNbDays'] < 28 and monthly_tot.iloc[0]['MONTH'] == 2) or
        (monthly_tot.iloc[0]['EffectiveNbDays'] < 30 and monthly_tot.iloc[0]['MONTH'] != 2)
        ):
            monthly_tot = monthly_tot.iloc[1:]
            
        demand_item_fct = pd.merge(demand_item_fct, monthly_tot[['YEAR','MONTH']], on=['YEAR','MONTH'], how='inner', sort=False)
        last_date = demand_item_fct['PERFORM_DATE'].max()
        last_date_hist = demand_item_fct['PERFORM_DATE'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            end=last_date + pd.DateOffset(months=horizon),
            freq='D'
        )
        future_rows = [{'ITEM_NUMBER_SITE': item_id, 'PERFORM_DATE': date} for date in future_dates]
        future_df = pd.DataFrame(future_rows)
        demand_item_fct = pd.concat([demand_item_fct, future_df], ignore_index=True, sort=False)
        demand_item_fct.reset_index(drop=True, inplace=True)   

        last_year = demand_item_fct['PERFORM_DATE'].max().year
        last_month = demand_item_fct['PERFORM_DATE'].max().month        
        num_days_in_month = calendar.monthrange(last_year, last_month)[1]
        all_days = pd.date_range(
            start=pd.Timestamp(year=last_year, month=last_month, day=1),
            end=pd.Timestamp(year=last_year, month=last_month, day=num_days_in_month),
            freq='D'
        )
        demand_item_fct['YEAR'] = demand_item_fct['PERFORM_DATE'].dt.year
        demand_item_fct['MONTH'] = demand_item_fct['PERFORM_DATE'].dt.month        
        existing_days = demand_item_fct[
            (demand_item_fct['YEAR'] == last_year) & (demand_item_fct['MONTH'] == last_month)
        ]['PERFORM_DATE']
        missing_days = set(all_days) - set(existing_days)
        
        if missing_days:
            missing_rows = [{'ITEM_NUMBER_SITE': item_id, 'PERFORM_DATE': date} for date in sorted(missing_days)]
            missing_df = pd.DataFrame(missing_rows)
            demand_item_fct = pd.concat([demand_item_fct, missing_df], ignore_index=True, sort=False)
            demand_item_fct.sort_values('PERFORM_DATE', inplace=True)
            demand_item_fct.reset_index(drop=True, inplace=True)
            demand_item_fct['YEAR'] = demand_item_fct['PERFORM_DATE'].dt.year
            demand_item_fct['MONTH'] = demand_item_fct['PERFORM_DATE'].dt.month        

        #cut_off = pd.to_datetime('2025-09-07 00:00:00', format=ts_format_2)
        #demand_item_fct = demand_item_fct[demand_item_fct['PERFORM_DATE'] <= cut_off].copy()
    
        days = demand_item_fct[['PERFORM_DATE']].copy()
        days['ITEM_NUMBER_SITE'] = item_id
        days['DayOfYear'] = days['PERFORM_DATE'].dt.dayofyear
        rbf = RepeatingBasisFunction(n_periods=12,
                                            column="DayOfYear",
                                            input_range=(1,365),
                                            remainder="drop")
        rbf.fit(days)
        X_sup = pd.DataFrame(index=days.index, data=rbf.transform(days))
        X_sup = X_sup.add_prefix("M_")
        days['DayOfMonth'] = days['PERFORM_DATE'].dt.day
        days['DayOfMonth_sin'] = np.sin(2 * np.pi * days['DayOfMonth'] / 31)
        days['DayOfMonth_cos'] = np.cos(2 * np.pi * days['DayOfMonth'] / 31)
        if actual_category == "Daily":
            days['DayOfWeek'] = days['PERFORM_DATE'].dt.dayofweek
            X_sup = pd.merge(X_sup, days[['DayOfWeek']], left_index=True, right_index=True)    
        X_sup = pd.merge(X_sup, days[['DayOfMonth_sin', 'DayOfMonth_cos']], left_index=True, right_index=True)
        
        demand_item_fct['MonthPeriod'] = demand_item_fct['PERFORM_DATE'].dt.to_period('M')
        demand_item_fct['FirstDayOfMonth'] = demand_item_fct['MonthPeriod'].dt.start_time
        demand_item_fct['WEEK'] = ((demand_item_fct['PERFORM_DATE'] - demand_item_fct['FirstDayOfMonth']).dt.days // 7) + 1
        demand_item_fct['WEEK'] = demand_item_fct['WEEK'].astype('Int64')
        demand_item_fct.drop(['MonthPeriod', 'FirstDayOfMonth'], axis=1, inplace=True)
        demand_item_fct['EffectiveNbDays'] = 1
        
        first_future_date = demand_item_fct['PERFORM_DATE'].max() - pd.DateOffset(months=6) + pd.Timedelta(days=1)
        first_future_month = first_future_date.to_period('M')        
        mask_future = demand_item_fct['PERFORM_DATE'] > last_date
        
        demand_item_fct.loc[mask_future, 'Week2'] = (
            (demand_item_fct.loc[mask_future, 'PERFORM_DATE'].dt.day - 1) // 7 + 1
        )
        
        demand_item_fct['Week2'] = demand_item_fct['Week2'].astype('Int64')
        
        future_months = (demand_item_fct.loc[mask_future, ['YEAR', 'MONTH']].drop_duplicates()
                        .sort_values(['YEAR', 'MONTH'])
                        .reset_index(drop=True))        
        next_weeks = {}
        for _, row in future_months.iterrows():
            year, month = row['YEAR'], row['MONTH']
            weeks = demand_item_fct[
                (demand_item_fct['YEAR'] == year) & (demand_item_fct['MONTH'] == month) & mask_future
            ]['Week2'].unique()
            next_weeks[(year, month)] = sorted(weeks)
        
        
        forecast_data = demand_item_fct.copy()
        forecast_data = forecast_data[~pd.isna(forecast_data[demand_col])]
        for col in ['FORECASTED_DEMAND_INF', 'FORECASTED_DEMAND_SUP']:
            if col not in forecast_data.columns:
                forecast_data[col] = np.nan                
        forecast_data['History'] = 1
        cutoff_date = last_date - pd.Timedelta(days=364)  # 364 to include both endpoints
        last_year_data = forecast_data[forecast_data['PERFORM_DATE'] >= cutoff_date].copy()

        # === STRUCTURAL BREAK DETECTION AND ENSEMBLE FORECASTING ===
        
        # Detect structural breaks in the historical data
        break_points, break_strength, post_break_data = detect_structural_breaks(
            forecast_data, 
            date_col='PERFORM_DATE',
            demand_col=demand_col,
            min_break_size=90,
            cusum_threshold=2.5,
            variance_threshold=0.7
        )
        
        
        # Store break information for production parameter adjustment
        break_info = (break_points, break_strength, post_break_data)
        
        # Apply ensemble forecasting if significant breaks detected
        if break_strength > 0.3:
            
            # Generate ensemble forecast for the prediction horizon
            horizon_days = horizon * 30  # Convert months to approximate days
            ensemble_forecast, confidence_multiplier = ensemble_forecast_with_breaks(
                forecast_data, 
                post_break_data, 
                break_strength,
                date_col='PERFORM_DATE',
                demand_col=demand_col,
                forecast_horizon=horizon_days
            )
                        
            ensemble_forecast_result = ensemble_forecast
            confidence_multiplier_result = confidence_multiplier
            break_info_result = break_info
        else:
            ensemble_forecast_result = None
            confidence_multiplier_result = 1.0
            break_info_result = break_info        
        
        weekly_tot = demand_item_fct.groupby(['YEAR','MONTH','Week2']).agg({'EffectiveNbDays': 'sum', #'Day Name':'first',                                                                 
                                                                'PERFORM_DATE': ['min', 'max']}).reset_index()
        weekly_tot.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in weekly_tot.columns]
        weekly_tot.rename(columns={'PERFORM_DATE_min': 'PERFORM_DATE', 'EffectiveNbDays_sum': 'EffectiveNbDays'}, inplace=True)
        weekly_tot = weekly_tot.sort_values('PERFORM_DATE')

        weekly_kpis = forecast_data.groupby(['ITEM_NUMBER_SITE', 'YEAR','MONTH','WEEK']).agg({demand_col: 'sum'}).reset_index()
        
        weekly_forecast=[]
        weekly_best_params = []
        total_week_idx = 0
        for (year, month), weeks in next_weeks.items():
            for week_idx, week in enumerate(weeks):
                total_week_idx += 1
                print(f"Year: {year}, Month: {month}, Week: {week}")
                train_item = forecast_data.groupby(['YEAR', 'MONTH', 'WEEK']).agg({
                    'EffectiveDemand': 'sum',
                    'PERFORM_DATE': 'min',
                    demand_col: 'sum'
                }).reset_index()

                train_item = train_item.sort_values('PERFORM_DATE')
                y = train_item.set_index('PERFORM_DATE')['EffectiveDemand']
            
                alphas = [0.1, 0.2, 0.5, 0.6]
                betas = [0.1, 0.2, 0.5, 0.6]
    
                best_mae_es_d = np.inf
                best_params = None
                best_forecast = None
                split_idx = int(len(y) * 0.9)
                y_train, y_test = y[:split_idx], y[split_idx:]
                                
                if (y == 0).any():
                    for alpha in alphas:
                        for beta in betas:
                            try:
                                model = ExponentialSmoothing(
                                    y_train,
                                    trend='add',
                                    seasonal=None,
                                    initialization_method="estimated"
                                )
                                fit = model.fit(smoothing_level=alpha,
                                                smoothing_trend=beta,
                                                optimized=False)
                                forecast = fit.forecast(len(y_test))
                                mae_value_d = mean_absolute_error(y_test, forecast)
                                if mae_value_d < best_mae_es_d:
                                    best_mae_es_d = mae_value_d
                                    best_params = (alpha, beta)
                            except Exception as e:
                                continue
                    model_full = ExponentialSmoothing(
                        y,
                        trend='add',
                        seasonal=None,
                        initialization_method="estimated"
                    )
                    fit_full = model_full.fit(
                        smoothing_level=max(best_params[0], min(alphas)),
                        smoothing_trend=max(best_params[1], min(betas)),
                        optimized=False)
                    weekly_forecast = fit_full.forecast(1).values[0]
                    weekly_forecast = pd.DataFrame({'Forecasted Effective Demand': [weekly_forecast]})

                else:
                    for alpha in alphas:
                        for beta in betas:
                            try:                   
                                model = ExponentialSmoothing(
                                    y_train,
                                    trend='mul',
                                    seasonal=None,
                                    initialization_method="estimated"
                                )
                                fit = model.fit(smoothing_level=alpha,
                                                smoothing_trend=beta,
                                                optimized=False)
                                forecast = fit.forecast(len(y_test))
                                mae_value_d = mean_absolute_error(y_test, forecast)
                                if mae_value_d < best_mae_es_d:
                                    best_mae_es_d = mae_value_d
                                    best_params = (alpha, beta)
                            except Exception as e:
                                continue
                    model_full = ExponentialSmoothing(
                        y,
                        trend='mul',
                        seasonal=None,
                        initialization_method="estimated"
                    )
                    fit_full = model_full.fit(
                        smoothing_level=max(best_params[0], min(alphas)),
                        smoothing_trend=max(best_params[1], min(betas)),
                        optimized=False
                    )
                    weekly_forecast = fit_full.forecast(1).values[0]
                    weekly_forecast = pd.DataFrame({'Forecasted Effective Demand': [weekly_forecast]})

                demand_days_es = np.ceil(weekly_forecast['Forecasted Effective Demand'].iloc[0])

                y.reset_index(drop=True, inplace=True)
                ts_train_item = TimeSeries.from_dataframe(y.to_frame())
                base_lags = [1, 2, 3, 4, 5, 6, 7, 12, 14]
                if total_week_idx <= 4:
                    lags_to_test = base_lags
                elif 5 <= total_week_idx <= 8:
                    lags_to_test = base_lags + [21, 28]
                elif 9 <= total_week_idx <= 12:
                    lags_to_test = base_lags + [21, 28, 35]
                else:
                    lags_to_test = base_lags + [21, 28, 35, 42]            

                best_lag = None
                best_mae_lr_d = float('inf')           
                split_idx = int(len(ts_train_item) * 0.9)
                train_ts = ts_train_item[:split_idx]
                val_ts = ts_train_item[split_idx:]
                
                for lag in lags_to_test:
                    model = LinearRegressionModel(lags=[-lag])
                    model.fit(train_ts)
                    pred = model.predict(len(val_ts))
                    error = mae(val_ts, pred)
                    if error < best_mae_lr_d:
                        best_mae_lr_d = error
                        best_lag = lag
                
                model = LinearRegressionModel(lags=[-best_lag])
                model.fit(ts_train_item)
                demand_days_lr = model.predict(1)
                demand_days_lr = int(np.round(demand_days_lr.values()[0, 0]))
                if best_mae_es_d > 0 and best_mae_lr_d > 0:
                    wgt1 = (1/best_mae_es_d) / (1/best_mae_es_d + 1/best_mae_lr_d)
                    wgt2 = (1/best_mae_lr_d) / (1/best_mae_es_d + 1/best_mae_lr_d)
                    demand_days = int(np.round(wgt1 * demand_days_es + wgt2 * demand_days_lr))
                elif best_mae_es_d <= 0 and best_mae_lr_d > 0:
                    demand_days = int(demand_days_lr)
                elif best_mae_es_d > 0 and best_mae_lr_d <= 0:
                    demand_days = int(demand_days_es)
                else:
                    demand_days = 0
            
                simulation_period_tsfresh = 14
                week_row = weekly_tot[
                    (weekly_tot['YEAR'] == year) &
                    (weekly_tot['MONTH'] == month) &
                    (weekly_tot['Week2'] == week)
                ]
                if not week_row.empty:
                    simulation_period_days = week_row.iloc[0]['EffectiveNbDays']
                else:
                    simulation_period_days = 7 
                simulation_period_days_prev = int(monthly_tot.iloc[-1]['DaysInMonth'])
                
                demand_days = min(demand_days, simulation_period_days)
                weekly_forecast['Forecasted Effective Demand'] = [demand_days]                

                if week==1:
                    train_item_month = train_item.groupby(['YEAR', 'MONTH']).agg({
                        'PERFORM_DATE': 'min',
                        demand_col: 'max'
                    }).reset_index()
                    train_item_month = train_item_month.sort_values('PERFORM_DATE')

                    y = train_item_month.set_index('PERFORM_DATE')[demand_col].asfreq('MS')
                    warning_raised = False
                    
                    alphas = [0.2, 0.25, 0.35, 0.5, 0.6]
                    betas = [0.1, 0.2, 0.35, 0.5, 0.6]
                    best_mae_es = 0 # AF adding bc otherwise doesn't exist. Thus, 0 if demand_days_es = 0
                    if demand_days_es > 0:    
                        best_mae_es = np.inf
                        best_params = None
                        best_forecast = None
                        split_idx = int(len(y) * 0.7)
                        y_train, y_test = y[:split_idx], y[split_idx:]
    
                        if (y == 0).any():
                            for alpha in alphas:
                                for beta in betas:
                                    try:
                                        model = ExponentialSmoothing(
                                            y_train,
                                            trend='add',
                                            seasonal=None,
                                            initialization_method="estimated"
                                        )
                                        fit = model.fit(smoothing_level=max(alpha, min(alphas)),
                                                        smoothing_trend=max(beta, min(betas)),
                                                        optimized=False)
                                        forecast = fit.forecast(len(y_test))
                                        mae_value = mean_absolute_error(y_test, forecast)
                                        if mae_value < best_mae_es:
                                            best_mae_es = mae_value
                                            best_params = (alpha, beta)
                                    except Exception as e:
                                        continue
            
                            model_full = ExponentialSmoothing(
                                y,
                                trend='add',
                                seasonal=None,
                                initialization_method="estimated"
                            )
                            fit_full = model_full.fit(
                                smoothing_level=max(best_params[0], min(alphas)),
                                smoothing_trend=max(best_params[1], min(betas)),
                                optimized=False
                            )
                            monthly_forecast_max_es = fit_full.forecast(1).values[0]
                    
                        else:
                            for alpha in alphas:
                                for beta in betas:
                                    try:
                                        model = ExponentialSmoothing(
                                            y_train,
                                            trend='mul',
                                            seasonal=None,
                                            initialization_method="estimated"
                                        )
                                        fit = model.fit(smoothing_level=max(alpha, min(alphas)),
                                                        smoothing_trend=max(beta, min(betas)),
                                                        optimized=False)
                                        forecast = fit.forecast(len(y_test))
                                        mae_value = mean_absolute_error(y_test, forecast)
                                        if mae_value < best_mae_es:
                                            best_mae_es = mae_value
                                            best_params = (alpha, beta)
                                    except Exception as e:
                                        continue
            
                            model_full = ExponentialSmoothing(
                                y,
                                trend='mul',
                                seasonal=None,
                                initialization_method="estimated"
                            )
                            fit_full = model_full.fit(
                                smoothing_level=max(best_params[0], min(alphas)),
                                smoothing_trend=max(best_params[1], min(betas)),
                                optimized=False
                            )
                            monthly_forecast_max_es = fit_full.forecast(1).values[0]
                    else:
                        monthly_forecast_max_es = 0
                    
                    y = train_item_month.set_index('PERFORM_DATE')[demand_col]
                    y = y.asfreq('MS')
                    ts_train_item = TimeSeries.from_dataframe(y.to_frame())
                    base_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    if total_week_idx <= 15:
                        lags_to_test = base_lags
                    elif 16 <= total_week_idx <= 40:
                        lags_to_test = base_lags + [10, 12]
                    elif 41 <= total_week_idx <= 55:
                        lags_to_test = base_lags + [10, 12, 18]
                    else:
                        lags_to_test = base_lags + [10, 12, 18, 24]          
                    best_lag = None
                    best_mae_lr = float('inf')           
                    split_idx = int(len(ts_train_item) * 0.8)
                    train_ts = ts_train_item[:split_idx]
                    val_ts = ts_train_item[split_idx:]
                    
                    for lag in lags_to_test:
                        model = LinearRegressionModel(lags=[-lag])
                        model.fit(train_ts)
                        pred = model.predict(len(val_ts))
                        error = mae(val_ts, pred)
                        if error < best_mae_lr:
                            best_mae_lr = error
                            best_lag = lag
                    
                    model = LinearRegressionModel(lags=[-best_lag])
                    model.fit(ts_train_item)
                    monthly_forecast_max_lr = model.predict(1)
                    monthly_forecast_max_lr = float(monthly_forecast_max_lr.values()[0, 0])
    
                    if best_mae_es > 0 and best_mae_lr > 0:
                        if monthly_forecast_max_es > 0 and monthly_forecast_max_lr > 0:
                            wgt1 = (1/best_mae_es) / (1/best_mae_es + 1/best_mae_lr)
                            wgt2 = (1/best_mae_lr) / (1/best_mae_es + 1/best_mae_lr)
                            monthly_forecast_max = int(np.round(wgt1 * monthly_forecast_max_es + wgt2 * monthly_forecast_max_lr))
                        elif monthly_forecast_max_es >=0 and monthly_forecast_max_lr < 0:
                            monthly_forecast_max = int(monthly_forecast_max_es) 
                        elif monthly_forecast_max_es < 0 and monthly_forecast_max_lr >= 0:
                            monthly_forecast_max = int(monthly_forecast_max_lr) 
                    else:
                        monthly_forecast_max = 0
            
                demand_y = forecast_data[['PERFORM_DATE', demand_col]].copy()
                demand_X = forecast_data.copy()
                demand_y = demand_y[~pd.isna(demand_y[demand_col])]
                demand_X = demand_X[~pd.isna(demand_X[demand_col])]

                next_days = pd.DataFrame([demand_X['PERFORM_DATE'].iloc[-1] + timedelta(days=i) for i in range(1, simulation_period_days+1)])   
                next_days = next_days.rename(columns = {0:'PERFORM_DATE'})
                demand_X.drop(columns=['YEAR','MONTH','DaysInMonth','EffectiveNbDays'], inplace=True, errors='ignore')            
                
                df = demand_X[['PERFORM_DATE',demand_col]].copy()
                df['PERFORM_DATE'] = df['PERFORM_DATE'].apply(str)
                df['Duplic'] = demand_col
                Tri = 'PERFORM_DATE'
                nb_months = 1
                max_nb_months = 12
                while df[demand_col].iloc[-nb_months * simulation_period_days_prev:].nunique() == 1 and nb_months < max_nb_months:
                    print(f"Validation period contains only identical values. Expanding to {nb_months + 1} months...")
                    nb_months += 1
                print(f"nb_months :{nb_months}")
                def parse_weekly_feature_name_to_tsfresh_config(feature_name):
                    """
                    Convert a tsfresh feature name to its configuration parameters.
                    Example: 'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"'
                    Returns: ('agg_linear_trend', {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'})
                    """
                    # Remove the column prefix
                    if feature_name.startswith('ACTUAL_DEMAND__'):
                        feature_name = feature_name[len('ACTUAL_DEMAND__'):]
                    
                    # Split by double underscore
                    parts = feature_name.split('__')
                    if not parts:
                        return None, None
                    
                    feature_func = parts[0]
                    params = {}
                    
                    # Parse parameters
                    for part in parts[1:]:
                        # Clean quotes from parameter values
                        clean_part = part.strip('"')
                        
                        if clean_part.startswith('attr_'):
                            params['attr'] = clean_part[5:].strip('"')
                        elif clean_part.startswith('chunk_len_'):
                            params['chunk_len'] = int(clean_part[10:])
                        elif clean_part.startswith('f_agg_'):
                            params['f_agg'] = clean_part[6:].strip('"')
                        elif clean_part.startswith('coeff_'):
                            params['coeff'] = int(clean_part[6:])
                        elif clean_part.startswith('q_'):
                            params['q'] = float(clean_part[2:])
                        elif clean_part.startswith('maxlag_'):
                            params['maxlag'] = int(clean_part[7:])
                        elif clean_part.startswith('num_segments_'):
                            params['num_segments'] = int(clean_part[13:])
                        elif clean_part.startswith('segment_focus_'):
                            params['segment_focus'] = int(clean_part[14:])
                        elif clean_part.startswith('isabs_'):
                            params['isabs'] = clean_part[6:] == 'True'
                        elif clean_part.startswith('qh_'):
                            params['qh'] = float(clean_part[3:])
                        elif clean_part.startswith('ql_'):
                            params['ql'] = float(clean_part[3:])
                        elif clean_part.startswith('w_'):
                            params['w'] = int(clean_part[2:])
                        elif clean_part.startswith('widths_'):
                            # Parse tuple like "(2, 5, 10, 20)"
                            widths_str = clean_part[7:].strip('()')
                            params['widths'] = tuple(map(int, widths_str.split(', ')))
                        elif clean_part.startswith('aggtype_'):
                            params['aggtype'] = clean_part[8:].strip('"')
                        elif clean_part.startswith('autolag_'):
                            params['autolag'] = clean_part[8:].strip('"')
                        elif clean_part.startswith('number_of_maxima_'):
                            params['number_of_maxima'] = int(clean_part[17:])
                        elif clean_part.startswith('max_bins_'):
                            params['max_bins'] = int(clean_part[9:])
                        elif clean_part.startswith('normalize_'):
                            params['normalize'] = clean_part[10:] == 'True'
                        elif clean_part.startswith('bins_'):
                            params['bins'] = int(clean_part[5:])
                        elif clean_part.startswith('r_'):
                            # Handle both int and float values for r parameter
                            try:
                                params['r'] = int(clean_part[2:])
                            except ValueError:
                                params['r'] = float(clean_part[2:])
                        elif clean_part.startswith('t_'):
                            # Handle both int and float values for t parameter  
                            try:
                                params['t'] = int(clean_part[2:])
                            except ValueError:
                                params['t'] = float(clean_part[2:])
                        elif clean_part.startswith('k_'):
                            params['k'] = int(clean_part[2:])
                        elif clean_part.startswith('n_'):
                            params['n'] = int(clean_part[2:])
                    
                    return feature_func, params if params else None
                
                def create_tsfresh_config_from_feature_list(feature_list):
                    """
                    Create a tsfresh configuration dictionary from a list of feature names.
                    """
                    config = {}
                    
                    for feature_name in feature_list:
                        func_name, params = parse_weekly_feature_name_to_tsfresh_config(feature_name)
                        if func_name:
                            if func_name not in config:
                                config[func_name] = [] if params else None
                            
                            if params and config[func_name] is not None:
                                if params not in config[func_name]:
                                    config[func_name].append(params)
                            elif params is None:
                                config[func_name] = None
                    
                    return config
                
                predefined_features_w_14 = create_tsfresh_config_from_feature_list(features_w_14_all)
                predefined_features_prev_M = create_tsfresh_config_from_feature_list(features_prev_M_all) 
                predefined_features_prev_6M = create_tsfresh_config_from_feature_list(features_prev_6M_all)
                
                rolling_configs = [
                    (nb_months * simulation_period_days_prev, "_prev_M", predefined_features_prev_M),
                    (6 * nb_months * simulation_period_days_prev, "_prev_6M", predefined_features_prev_6M),
                    (nb_months * simulation_period_tsfresh, "_w14", predefined_features_w_14),
                    ]
                
                feats = []
                for max_timeshift, suffix, feature_settings in rolling_configs:
                    rolled = roll_time_series(df, "Duplic", Tri, max_timeshift=max_timeshift, n_jobs=1)
                    feat = extract_features(
                        rolled.drop("Duplic", axis=1),
                        column_id="id",
                        column_sort=Tri,
                        default_fc_parameters=feature_settings,
                        impute_function=impute,
                        n_jobs=1
                    )
                    feat = feat.set_index(feat.index.map(lambda x: x[1]), drop=True)
                    feat.columns = feat.columns.str.replace('[",#,@,&,.]', '')
                    feat = feat.add_suffix(suffix)
                    feat.reset_index(drop=False, inplace=True)
                    feat['index'] = feat['index'].astype('datetime64[ns]')
                    feat.rename(columns={'index': 'PERFORM_DATE'}, inplace=True)
                    del rolled
                    gc.collect()
                    
                    if feat.shape[0] < df.shape[0]:
                        df['PERFORM_DATE'] = df['PERFORM_DATE'].astype('datetime64[ns]')
                        feat = pd.merge(feat, df[['PERFORM_DATE']], on=['PERFORM_DATE'], how='right', sort=False)
                        feat.set_index('PERFORM_DATE', inplace=True)
                        if feat.isnull().sum().sum() > 0:
                            feat = feat.interpolate(method='linear')
                    else:
                        feat.set_index('PERFORM_DATE', inplace=True)
                    feats.append(feat)
                feat = pd.concat(feats, axis=1)                
                                
                demand_y_select = pd.merge(demand_y, demand_X[['PERFORM_DATE']], on=['PERFORM_DATE'], how='inner', sort=False)
                demand_y_select = demand_y_select.iloc[:-simulation_period_days_prev]
                demand_y_select = demand_y_select.set_index('PERFORM_DATE')
                
                correlation_data = pd.merge(demand_y_select, feat, left_index=True, right_index=True, sort= False)
                correlation_matrix = correlation_data.corr()
                correlation_matrix = correlation_matrix.iloc[:,:1].squeeze()
                selected_columns = correlation_matrix.abs().sort_values(ascending=False).index
                selected_columns = selected_columns.drop(demand_col, errors='ignore')
                selected_columns = selected_columns[:65]
                week_features = pd.DataFrame({
                    'feature': selected_columns,
                    'year': year,
                    'month': month, 
                    'week': week,
                    'total_week_idx': total_week_idx,
                    'feature_rank': range(1, len(selected_columns) + 1)
                })
                
                all_best_feats.append(week_features)
                
                feat = feat[selected_columns]
                
                correlation_matrix = feat.corr()
                perfect_corr = []
                for col in correlation_matrix.columns:
                    for row_idx in correlation_matrix.index:
                        if col != row_idx and correlation_matrix.loc[row_idx, col] == 1:
                            perfect_corr.append((row_idx, col))
                perfect_corr_groups = []
                checked = set()
                for col in correlation_matrix.columns:
                    group = [col]
                    for other_col in correlation_matrix.columns:
                        if col != other_col and correlation_matrix.loc[col, other_col] == 1:
                            group.append(other_col)
                    group = sorted(set(group), key=lambda x: list(feat.columns).index(x))
                    if len(group) > 1 and not any(c in checked for c in group):
                        perfect_corr_groups.append(group)
                        checked.update(group)
                
                cols_to_drop = []
                for group in perfect_corr_groups:
                    cols_to_drop.extend(group[1:])
                feat = feat.drop(columns=cols_to_drop)
                
                rolling_windows = [3, 7, 14, 28, 56]            
                additional_feat = add_rolling_features(
                demand_X, 
                demand_col=demand_col, 
                windows=rolling_windows)
                additional_feat.set_index('PERFORM_DATE', inplace=True)
                additional_feat.drop(columns={'ACTUAL_DEMAND_rollskew_3', 'ACTUAL_DEMAND_rollkurt_3', 'ACTUAL_DEMAND_rollskew_7', 'ACTUAL_DEMAND_rollkurt_7'}, inplace=True)
                col_name='ACTUAL_DEMAND_rollmean_56'
                nan_dates = additional_feat.index[additional_feat[col_name].isna()]
                if len(nan_dates) > 0:
                    min_nan_date = nan_dates.min()
                    max_nan_date = nan_dates.max()
                    additional_feat = additional_feat[~additional_feat.index.duplicated(keep='first')]
                    mask_nan_period = (additional_feat.index >= min_nan_date) & (additional_feat.index <= max_nan_date)
                    period_offset = pd.DateOffset(years=1)
                    nan_period_dates = additional_feat.index[mask_nan_period]
                    corresponding_dates = nan_period_dates + period_offset
                    valid_mask = corresponding_dates.isin(additional_feat.index)
                    nan_period_dates_valid = nan_period_dates[valid_mask]
                    corresponding_dates_valid = corresponding_dates[valid_mask]
                    for col in additional_feat.columns:
                        nan_mask = additional_feat.loc[nan_period_dates_valid, col].isna()
                        idx_to_fill = nan_period_dates_valid[nan_mask]
                        idx_source = corresponding_dates_valid[nan_mask]
                        additional_feat.loc[idx_to_fill, col] = additional_feat.loc[idx_source, col].values
                
                additional_feat = additional_feat.sort_index()
                additional_feat = additional_feat[~additional_feat.index.duplicated(keep='first')]
                additional_feat = additional_feat.interpolate(method='spline', order=2, axis=0)
                additional_feat = additional_feat.clip(lower=0)
                additional_feat = additional_feat.loc[:, additional_feat.nunique(dropna=False) > 1]
                
                target = demand_y_select.squeeze()

                new_cols = {}
                for col1 in additional_feat.columns:
                    for col2 in feat.columns:
                        new_col_name = f"{col1}_x_{col2}"
                        product = additional_feat[col1] * feat[col2]
                        corr = product.corr(target)
                        if pd.notna(corr) and abs(corr) >= 0.98:
                            new_cols[new_col_name] = product

                if new_cols:
                    new_cols_df = pd.DataFrame(new_cols, index=additional_feat.index)
                    additional_feat = pd.concat([additional_feat, new_cols_df], axis=1)
                feat = feat[~feat.index.duplicated(keep='first')]
                additional_feat = additional_feat[~additional_feat.index.duplicated(keep='first')]
                feat = pd.concat([feat, additional_feat], axis=1)
                feat = feat.dropna(axis=1)

                correlation_data = pd.merge(demand_y[demand_col], feat, left_index=True, right_index=True, sort= False)
                correlation_matrix = correlation_data.corr()
                correlation_matrix = correlation_matrix.iloc[:,:1].squeeze()
                selected_columns = correlation_matrix.abs().sort_values(ascending=False).index
                selected_columns = selected_columns.drop(demand_col, errors='ignore')
                selected_columns = selected_columns[:75]
                feat = feat[selected_columns]
                        
                imp_scaler = StandardScaler()
                feat_sc = imp_scaler.fit_transform(feat)
                
                feat_sc_next_days = []
                time_index = pd.DatetimeIndex(feat.index)
                for column in feat_sc.T:
                    feat_sc_next_days.append(TimeSeries.from_times_and_values(time_index, column))   
                
                series_names = np.arange(feat_sc.shape[1])
                preds = list()
                initial_dim_x = 8
                def _kalman_fit_predict(args):
                    i, series, dim_x, horizon = args
                    model = KalmanForecaster(dim_x=dim_x)
                    model.fit(series=series)
                    p = model.predict(horizon).to_dataframe().reset_index()
                    p[i] = series_names[i]
                    return i, p

                def _kalman_fallback(args):
                    i, series, horizon = args
                    last_val = series.values()[-1] if len(series.values()) > 0 else 0
                    p = pd.DataFrame({
                        'time': pd.date_range(start=series.end_time() + pd.Timedelta(days=1),
                                              periods=horizon, freq='D'),
                        '0': [last_val] * horizon
                    })
                    p[i] = series_names[i]
                    return i, p

                successful_forecast = False
                for dim_x in range(initial_dim_x, 0, -1):  # Try from initial_dim_x down to 1
                    try:
                        temp_preds = list()
                        for i, series in enumerate(feat_sc_next_days):
                            model = KalmanForecaster(dim_x=dim_x)
                            model.fit(series=series)
                            p = model.predict(simulation_period_days).to_dataframe().reset_index()
                            p[i] = series_names[i]
                            temp_preds.append(p)
                        preds = temp_preds  # Only assign if all series processed successfully
                        successful_forecast = True
                        break
                    except (np.linalg.LinAlgError, ValueError) as e:
                        if dim_x == 1:
                            # If even dim_x=1 fails, use a simple fallback
                            preds = []
                            for i, series in enumerate(feat_sc_next_days):
                                # Create a simple forecast using the last value
                                last_val = series.values()[-1] if len(series.values()) > 0 else 0
                                p = pd.DataFrame({
                                    'time': pd.date_range(start=series.end_time() + pd.Timedelta(days=1), 
                                                         periods=simulation_period_days, freq='D'),
                                    '0': [last_val] * simulation_period_days
                                })
                                p[i] = series_names[i]
                                preds.append(p)
                            break
                        continue  # Try next lower dim_x value
                
                preds = pd.concat(preds, axis=0, ignore_index=True)
                preds = preds.iloc[:, 1].to_frame()
                feat_next_days = pd.DataFrame()
                num_columns = int(preds.shape[0]/simulation_period_days)
                column_data_list = []
                for i in range(num_columns):
                    start_index = i * simulation_period_days
                    end_index = (i + 1) * simulation_period_days
                    column_data = preds['0'].iloc[start_index:end_index].reset_index(drop=True)
                    column_data_list.append(column_data)
                feat_next_days = pd.concat(column_data_list, axis=1).to_numpy()
                feat_next_days = np.concatenate((feat_sc, feat_next_days), axis = 0)
                feat_next_days = pd.DataFrame(imp_scaler.inverse_transform(feat_next_days))
                    
                feat_next_days.columns = feat.columns
                
                for col in feat_next_days.columns:
                    pred_idx = np.arange(len(feat_next_days))[-simulation_period_days:]
                    ref_idx = np.arange(len(feat_next_days))[-(365):-simulation_period_days]
                
                    pred_std = feat_next_days.loc[pred_idx, col].std()
                    ref_std = feat_next_days.loc[ref_idx, col].std()
                
                    if pred_std < ref_std and pred_std > 0:
                        pred_mean = feat_next_days.loc[pred_idx, col].mean()
                        scaling_factor = ref_std / pred_std
                        feat_next_days.loc[pred_idx, col] = pred_mean + (feat_next_days.loc[pred_idx, col] - pred_mean) * scaling_factor
                    elif pred_std == 0 and ref_std > 0:
                        ref_mean = feat_next_days.loc[ref_idx, col].mean()
                        feat_next_days.loc[pred_idx, col] = ref_mean
                
                train = feat_next_days.iloc[:-simulation_period_days]
                test = feat_next_days.iloc[-simulation_period_days:]
                cols_to_clip = train.columns[train.max() == 1]
                feat_next_days.loc[feat_next_days.index[-simulation_period_days:], cols_to_clip] = test[cols_to_clip].clip(upper=1)
                
                col_name = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0_w14'
                ref_col = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4_w14'
                if col_name in feat_next_days.columns and ref_col in feat_next_days.columns:   
                    col_idx = feat_next_days.columns.get_loc(col_name)
                    ref_idx = feat_next_days.columns.get_loc(ref_col)
                    mask = feat_next_days.iloc[-simulation_period_days:, ref_idx] == 0.0
                    feat_next_days.iloc[-simulation_period_days:, col_idx] = np.where(
                        mask,
                        0.0,
                        feat_next_days.iloc[-simulation_period_days:, col_idx]
                    )

                col_name = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0_prev_M'
                ref_col = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4_prev_M'
                if col_name in feat_next_days.columns and ref_col in feat_next_days.columns:   
                    col_idx = feat_next_days.columns.get_loc(col_name)
                    ref_idx = feat_next_days.columns.get_loc(ref_col)
                    mask = feat_next_days.iloc[-simulation_period_days:, ref_idx] == 0.0
                    feat_next_days.iloc[-simulation_period_days:, col_idx] = np.where(
                        mask,
                        0.0,
                        feat_next_days.iloc[-simulation_period_days:, col_idx]
                    )

                col_name = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0_prev_6M'
                ref_col = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4_prev_6M'
                if col_name in feat_next_days.columns and ref_col in feat_next_days.columns:   
                    col_idx = feat_next_days.columns.get_loc(col_name)
                    ref_idx = feat_next_days.columns.get_loc(ref_col)
                    mask = feat_next_days.iloc[-simulation_period_days:, ref_idx] == 0.0
                    feat_next_days.iloc[-simulation_period_days:, col_idx] = np.where(
                        mask,
                        0.0,
                        feat_next_days.iloc[-simulation_period_days:, col_idx]
                    )
    
                for col in feat_next_days.columns:
                    pred_idx = np.arange(len(feat_next_days))[-simulation_period_days:]
                    ref_idx = np.arange(len(feat_next_days))[:-simulation_period_days]
                
                    pred_vals = feat_next_days.loc[pred_idx, col]
                    ref_min = feat_next_days.loc[ref_idx, col].min()
                
                    if ref_min >= 0:
                        mask = feat_next_days.loc[pred_idx, col] < 0
                        if mask.any():
                            feat_next_days.loc[pred_idx, col] = feat_next_days.loc[pred_idx, col].mask(mask, 0)
    
                for col in feat_next_days.columns:
                    pred_idx = np.arange(len(feat_next_days))[-simulation_period_days:]
                    ref_idx = np.arange(len(feat_next_days))[:-simulation_period_days]
                
                    pred_vals = feat_next_days.loc[pred_idx, col]
                    ref_max = feat_next_days.loc[ref_idx, col].max()
                
                    if ref_max < 0:
                        mask = feat_next_days.loc[pred_idx, col] > 0
                        if mask.any():
                            feat_next_days.loc[pred_idx, col] = feat_next_days.loc[pred_idx, col].mask(mask, 0)
    
                demand_X = pd.merge(demand_item_fct[['PERFORM_DATE']], feat_next_days, left_index=True, right_index=True, sort=False)
                demand_X = pd.merge(demand_X, X_sup, left_index=True, right_index=True, sort=False)           
                demand_X.set_index('PERFORM_DATE', inplace=True)
                demand_X = demand_X[~demand_X.index.duplicated(keep='first')]
                demand_X = reduce_mem_usage(demand_X)
                
                y_train = forecast_data[['PERFORM_DATE', demand_col]].copy()
                y_train.set_index('PERFORM_DATE', inplace=True)
                y_train = y_train[~y_train.index.duplicated(keep='first')]
                
                # Get structural break information from forecast_data attributes
                break_info = getattr(forecast_data, 'attrs', {}).get('break_info', None)
                base_weight = len(y_train) / np.count_nonzero(y_train)
                
                # Apply enhanced production parameters considering structural breaks
                production_params = get_production_parameters(
                    item_data=forecast_data,
                    break_info=break_info_result,
                    base_params={'weight': base_weight, 'regularization': 0.1}
                )
                # Use enhanced weight that considers structural break context
                wgt = production_params['weight']
                                
            
                X_test = demand_X.iloc[-simulation_period_days:].copy()
                X_train = demand_X.iloc[:-simulation_period_days].copy()
                X_train.index = demand_X.index[:-simulation_period_days]
                X_test.index = demand_X.index[-simulation_period_days:]
                day_pred = demand_X.index[-simulation_period_days:].to_frame(index=False)
                if nb_months > 1:
                    X_val = X_train[X_train.shape[0]-2*nb_months*simulation_period_days:X_train.shape[0]]
                else:
                    X_val = X_train[X_train.shape[0]-simulation_period_days_prev:X_train.shape[0]]
                
                if nb_months > 1:
                    X_train = X_train.iloc[:-2*nb_months*simulation_period_days,:]
                else:
                    X_train = X_train.iloc[:-simulation_period_days_prev,:]
        
                X_train = clean_feature_names(X_train)
                X_val = clean_feature_names(X_val)
                X_test = clean_feature_names(X_test)
                    
                y_val = y_train.merge(X_val.drop(X_val.columns, axis=1), left_index=True, right_index=True)
                y_train = y_train.merge(X_train.drop(X_train.columns, axis=1), left_index=True, right_index=True)            
                y_train = y_train.values.ravel()
                y_val = y_val.values.ravel()                
                spearman_corrs = {}
                for col in X_train.columns:
                    corr, _ = spearmanr(X_train[col], y_train)
                    spearman_corrs[col] = corr
                spearman_corrs_df = pd.DataFrame.from_dict(spearman_corrs, orient='index', columns=['spearman_corr'])            
                monotone_constraints = []
                for corr in spearman_corrs_df['spearman_corr']:
                    if corr > 0.71:
                        monotone_constraints.append(1)
                    elif corr < -0.7:
                        monotone_constraints.append(-1)
                    else:
                        monotone_constraints.append(0)            
                
                forecast = []
                
                sampler = optuna.samplers.TPESampler(seed=42)
                study = optuna.create_study(
                    study_name="lgbm", 
                    direction="minimize", 
                    pruner=optuna.pruners.HyperbandPruner(
                        min_resource=100,
                        max_resource=2000,
                        reduction_factor=3
                    )
                )                
                study.optimize(lambda trial: objective_weekly_LGBM_with_weight(
                trial,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                wgt=wgt,
                enhanced_regularization=production_params['regularization']),
                n_trials=1, callbacks=[print_callback]) # TODO: switch back to 30 trials
                best_params = study.best_params
                del study
                del sampler
                gc.collect()          
                
                '''
                best_params = {
                    'num_leaves': 137,
                    'max_depth': 10,
                    'learning_rate': 0.051838685,
                    'n_estimators': 1185,
                    'min_child_samples': 21,
                    'min_split_gain': 0.470949061,
                    'lambda_l1': 2.45655681,
                    'lambda_l2': 2.523018089,
                    'subsample': 0.754187641,
                    'colsample_bytree': 0.758028891,
                    'feature_fraction': 0.746900151,
                    'bagging_fraction': 0.761411114,
                    'bagging_freq': 5,
                    'max_bin': 59
                }
                '''
                X_full = pd.concat([X_train, X_val], axis=0)
                y_full = np.concatenate([y_train, y_val], axis=0)
                model = build_fit_weekly_LGBMRegressor(
                    X_full=X_full,
                    y_full=y_full,
                    monotone_constraints=monotone_constraints,
                    monotone_constraints_method="intermediate",
                    **best_params
                )
                pred = model.predict(X_test)
                best_params_df = pd.DataFrame([best_params])
                best_params_df['ITEM_NUMBER_SITE'] = item_id
                best_params_df['WEEK'] = week
                weekly_best_params.append(best_params_df)
                
                pct1 = len(X_train)/len(X_full)
                pct2 = simulation_period_days_prev/len(X_full)
                pct3 = len(X_val)/len(X_full)
                pct1 = pct1 - pct2
                
                (MX_train, MX_conformalize, MX_test,
                My_train, My_conformalize, My_test) = train_conformalize_test_split(
                    X_full, y_full,
                    train_size=pct1, conformalize_size=pct2, test_size=pct3,
                    random_state=1
                )
                confidence_level = 0.95
                mapie_regressor = SplitConformalRegressor(
                    estimator=model, confidence_level=confidence_level, prefit=True
                )
                mapie_regressor.conformalize(MX_conformalize, My_conformalize)
                y_pred, y_pred_interval = mapie_regressor.predict_interval(X_test)
                ci_05 = y_pred_interval[:, 0, 0]
                ci_95 = y_pred_interval[:, 1, 0]
                ci_05 = pd.DataFrame(ci_05, columns=['FORECASTED_DEMAND_INF'])
                pred = pd.DataFrame(pred, columns=[demand_col])
                ci_95 = pd.DataFrame(ci_95, columns=['FORECASTED_DEMAND_SUP'])
                        
                forecast = pd.concat([ci_05, pred, ci_95], axis=1)
                for col in forecast.columns:
                    forecast[col] = forecast[col].abs().round(0)
                forecast['FORECASTED_DEMAND_SUP'] = np.where(forecast[demand_col]==0,0,forecast['FORECASTED_DEMAND_SUP'])
                forecast['ITEM_NUMBER_SITE'] = item_id
                forecast = pd.merge(next_days, forecast, left_index=True, right_index=True, sort=False)
                demand_days = int(weekly_forecast['Forecasted Effective Demand'])
                        
                selected = []
                used = set()
                values = forecast[demand_col].copy()
    
                for _ in range(demand_days):
                    mask = np.ones(len(values), dtype=bool)
                    for idx in used:
                        mask[idx] = False
                        if idx > 0:
                            mask[idx-1] = False
                        if idx < len(values)-1:
                            mask[idx+1] = False
                    if not mask.any():
                        break
                    idx = values[mask].idxmax()
                    selected.append(idx)
                    used.add(idx)
    
                new_values = np.zeros(len(forecast), dtype=int)
                new_inf = np.zeros(len(forecast), dtype=int)
                new_sup = np.zeros(len(forecast), dtype=int)
                
                for idx in selected:
                    anchor_val = forecast.loc[idx, demand_col]
                    anchor_inf = forecast.loc[idx, 'FORECASTED_DEMAND_INF']
                    anchor_sup = forecast.loc[idx, 'FORECASTED_DEMAND_SUP']
                    if idx > 0 and idx-1 not in selected:
                        anchor_val += forecast.loc[idx-1, demand_col]
                        anchor_inf += forecast.loc[idx-1, 'FORECASTED_DEMAND_INF']
                        anchor_sup += forecast.loc[idx-1, 'FORECASTED_DEMAND_SUP']
                        new_values[idx-1] = 0
                        new_inf[idx-1] = 0
                        new_sup[idx-1] = 0
                    new_values[idx] = anchor_val
                    new_inf[idx] = anchor_inf
                    new_sup[idx] = anchor_sup
            
                forecast[demand_col] = new_values
                forecast['FORECASTED_DEMAND_INF'] = new_inf
                forecast['FORECASTED_DEMAND_SUP'] = new_sup                
                
                forecast['EffectiveDemand'] = np.where(forecast[demand_col]>0,1,0)
                forecast['YEAR'] = forecast['PERFORM_DATE'].dt.year
                forecast['MONTH'] = forecast['PERFORM_DATE'].dt.month
                forecast['WEEK'] = week
                forecast['History'] = 1
                forecast['EffectiveNbDays'] = 1
                forecast['Week2'] = np.nan
                if np.issubdtype(forecast['Week2'].dtype, np.floating):
                    forecast['Week2'] = forecast['Week2'].astype('Int64')
                forecast['DaysInMonth'] = forecast['PERFORM_DATE'].dt.days_in_month
                for col in forecast_data.columns:
                    if col not in forecast.columns:
                        forecast[col] = np.nan
                forecast = forecast[forecast_data.columns]
                forecast_data = pd.concat([forecast_data, forecast], axis=0)
                forecast_data.reset_index(drop=True, inplace=True)
                forecast_data.drop(columns={'EffectiveNbDays current week'}, inplace=True, errors='ignore')
                forecast_data[demand_col] = forecast_data[demand_col].round(0).astype(int)
                forecast_data['FORECASTED_DEMAND_INF'] = forecast_data['FORECASTED_DEMAND_INF'].round(0).astype('Int64')
                forecast_data['FORECASTED_DEMAND_SUP'] = forecast_data['FORECASTED_DEMAND_SUP'].round(0).astype('Int64')
                
                if week == 5:
                    week_sums = forecast_data[
                        (forecast_data['YEAR'] == year) &
                        (forecast_data['MONTH'] == month)
                    ].groupby('WEEK')[demand_col].sum()
                    max_weekly_value = week_sums.max()
                    max_week = week_sums.idxmax()
                    if max_weekly_value < monthly_forecast_max:
                        week_mask = (
                            (forecast_data['YEAR'] == year) &
                            (forecast_data['MONTH'] == month) &
                            (forecast_data['WEEK'] == max_week)
                        )
                        week_days = forecast_data[week_mask].index.tolist()

                        week_values = forecast_data.loc[week_days, demand_col]
                        max_day_value = week_values.max()
                        max_day_indices = week_values[week_values == max_day_value].index.tolist()

                        np.random.seed(RANDOM_SEED)
                        chosen_idx = np.random.choice(max_day_indices)
                        diff = monthly_forecast_max - max_weekly_value

                        forecast_data.at[chosen_idx, demand_col] += diff
                        forecast_data.at[chosen_idx, 'FORECASTED_DEMAND_INF'] += diff
                        forecast_data.at[chosen_idx, 'FORECASTED_DEMAND_SUP'] += diff
        
        forecast_data['FORECASTED_DEMAND_INF'] = np.minimum(
            forecast_data['FORECASTED_DEMAND_INF'],
            forecast_data['FORECASTED_DEMAND_SUP']
        )
        forecast_data['FORECASTED_DEMAND_SUP'] = np.maximum.reduce([
            forecast_data['FORECASTED_DEMAND_SUP'],
            forecast_data['FORECASTED_DEMAND_INF'],
            forecast_data[demand_col]
        ])
        forecast_data['EffectiveDemand'] = np.where(forecast_data['ACTUAL_DEMAND'] > 0,1,0)
        forecast_data['FORECASTED_DEMAND_INF'] = forecast_data['FORECASTED_DEMAND_INF'].astype(float)
        forecast_data['FORECASTED_DEMAND_SUP'] = forecast_data['FORECASTED_DEMAND_SUP'].astype(float)
        mask = (forecast_data['ACTUAL_DEMAND'] > 0) & (forecast_data['ACTUAL_DEMAND'] < forecast_data['FORECASTED_DEMAND_INF'])
        forecast_data.loc[mask, 'FORECASTED_DEMAND_INF'] = forecast_data.loc[mask, 'ACTUAL_DEMAND'] * 0.95
        forecast_data.loc[mask, 'FORECASTED_DEMAND_SUP'] = forecast_data.loc[mask, 'ACTUAL_DEMAND'] * 1.15
        forecast_data['FORECASTED_DEMAND_INF'] = forecast_data['FORECASTED_DEMAND_INF'].round(0)
        forecast_data['FORECASTED_DEMAND_SUP'] = forecast_data['FORECASTED_DEMAND_SUP'].round(0)
        if np.issubdtype(forecast_data['FORECASTED_DEMAND_INF'].dtype, np.floating):
            forecast_data['FORECASTED_DEMAND_INF'] = forecast_data['FORECASTED_DEMAND_INF'].astype('Int64')
        if np.issubdtype(forecast_data['FORECASTED_DEMAND_SUP'].dtype, np.floating):
            forecast_data['FORECASTED_DEMAND_SUP'] = forecast_data['FORECASTED_DEMAND_SUP'].astype('Int64')
        
        # Apply confidence multiplier if structural breaks were detected
        # Use the variables defined earlier instead of .attrs to avoid tsfresh conflicts
        confidence_multiplier = getattr(forecast_data, 'attrs', {}).get('confidence_multiplier', 1.0)
        break_info = getattr(forecast_data, 'attrs', {}).get('break_info', None)
        
        if confidence_multiplier_result > 1.0:
            
            # Expand confidence intervals around the central forecast
            central_forecast = forecast_data[demand_col]
            lower_diff = central_forecast - forecast_data['FORECASTED_DEMAND_INF']
            upper_diff = forecast_data['FORECASTED_DEMAND_SUP'] - central_forecast
            
            # Apply multiplier to the differences (expanding uncertainty)
            forecast_data['FORECASTED_DEMAND_INF'] = central_forecast - (lower_diff * confidence_multiplier_result)
            forecast_data['FORECASTED_DEMAND_SUP'] = central_forecast + (upper_diff * confidence_multiplier_result)
            
            # for non-negative intervals
            forecast_data['FORECASTED_DEMAND_INF'] = np.maximum(forecast_data['FORECASTED_DEMAND_INF'], 0)
            
        #all_best_params.append(pd.concat(weekly_best_params, ignore_index=True))

        true_forecast = forecast_data[forecast_data['PERFORM_DATE'] > last_date_hist].copy()
        global_interval = simulate_aggregate_intervals(true_forecast, dist='nbinom')
        global_interval['ITEM_NUMBER_SITE'] = item_id
        global_interval['Ratio1'] = global_interval['FORECASTED_DEMAND_INF'] / global_interval[demand_col]
        global_interval['Ratio2'] = global_interval['FORECASTED_DEMAND_SUP'] / global_interval[demand_col]
        last_date = forecast_data['PERFORM_DATE'].max()
        start_date = last_date - pd.DateOffset(months=horizon)
        yearly_tot = forecast_data.loc[
            (forecast_data['PERFORM_DATE'] > start_date) & (forecast_data['PERFORM_DATE'] <= last_date),
            demand_col
        ].sum()
        last_year = forecast_data.loc[
            (forecast_data['PERFORM_DATE'] > start_date) & (forecast_data['PERFORM_DATE'] <= last_date)
        ]

        global_interval['FORECASTED_DEMAND_INF'] = global_interval['Ratio1'] * yearly_tot
        global_interval['FORECASTED_DEMAND_SUP'] = global_interval['Ratio2'] * yearly_tot
        
        # Handle NaN and infinite values before converting to int
        global_interval['FORECASTED_DEMAND_INF'] = global_interval['FORECASTED_DEMAND_INF'].abs()
        global_interval['FORECASTED_DEMAND_SUP'] = global_interval['FORECASTED_DEMAND_SUP'].abs()
        
        # Replace NaN and infinite values with 0 before converting to int
        global_interval['FORECASTED_DEMAND_INF'] = global_interval['FORECASTED_DEMAND_INF'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        global_interval['FORECASTED_DEMAND_SUP'] = global_interval['FORECASTED_DEMAND_SUP'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        global_interval['Forecasted Demand'] = yearly_tot
        global_interval = global_interval[['ITEM_NUMBER_SITE', 'Forecasted Demand', 'FORECASTED_DEMAND_INF', 'FORECASTED_DEMAND_SUP']]

        forecast_results_summary = forecast_data.groupby(['ITEM_NUMBER_SITE', 'YEAR', 'MONTH', 'WEEK']).agg(
            Nb_Days_sum=('EffectiveDemand', 'sum'),
            Forecasted_Demand_sum=(demand_col, 'sum'),
            Forecasted_Demand_mean=(demand_col, 'mean'),
            Forecasted_Demand_std=(demand_col, 'std'),
            Forecasted_Demand_inf_sum=('FORECASTED_DEMAND_INF', 'sum'),
            Forecasted_Demand_inf_mean=('FORECASTED_DEMAND_INF', 'mean'),
            Forecasted_Demand_inf_std=('FORECASTED_DEMAND_INF', 'std'),
            Forecasted_Demand_sup_sum=('FORECASTED_DEMAND_SUP', 'sum'),
            Forecasted_Demand_sup_mean=('FORECASTED_DEMAND_SUP', 'mean'),
            Forecasted_Demand_sup_std=('FORECASTED_DEMAND_SUP', 'std'),
        ).reset_index()

        forecast_results_summary.rename(columns={
            'Nb_Days_sum': 'Number of Days of Demand',
            'Forecasted_Demand_sum': 'Forecasted Demand',
            'Forecasted_Demand_mean': 'Forecasted Demand mean',
            'Forecasted_Demand_std': 'Forecasted Demand std',
            'Forecasted_Demand_inf_sum': 'FORECASTED_DEMAND_INF',
            'Forecasted_Demand_inf_mean': 'FORECASTED_DEMAND_INF mean',
            'Forecasted_Demand_inf_std': 'FORECASTED_DEMAND_INF std',
            'Forecasted_Demand_sup_sum': 'FORECASTED_DEMAND_SUP',
            'Forecasted_Demand_sup_mean': 'FORECASTED_DEMAND_SUP mean',
            'Forecasted_Demand_sup_std': 'FORECASTED_DEMAND_SUP std',
        }, inplace=True)

        forecast_results_summary['Forecasted Demand mean'] = forecast_results_summary['Forecasted Demand mean'].round(3)
        forecast_results_summary['Forecasted Demand std'] = forecast_results_summary['Forecasted Demand std'].round(3)
        forecast_results_summary['FORECASTED_DEMAND_INF mean'] = forecast_results_summary['FORECASTED_DEMAND_INF mean'].round(3)
        forecast_results_summary['FORECASTED_DEMAND_INF std'] = forecast_results_summary['FORECASTED_DEMAND_INF std'].round(3)
        forecast_results_summary['FORECASTED_DEMAND_SUP mean'] = forecast_results_summary['FORECASTED_DEMAND_SUP mean'].round(3)
        forecast_results_summary['FORECASTED_DEMAND_SUP std'] = forecast_results_summary['FORECASTED_DEMAND_SUP std'].round(3)

        all_summaries.append(forecast_results_summary)
        all_global_interval.append(global_interval)
        all_forecasts.append(forecast_data)
        all_last_years.append(last_year)             
        all_best_params.extend(weekly_best_params)
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
            print("GPU memory cleared")
        except:
            pass
        try:
            del feat, demand_X, demand_y, feat_next_days
            del X_train, y_train, X_test, X_val
            del X_bt_train, X_bt_val, y_bt_train, y_bt_val
            del forecast_data, demand_item_fct
            print("Large variables cleared")
        except:
            pass        
        gc.collect()
    results = pd.concat(all_forecasts)[["ITEM_NUMBER_SITE","PERFORM_DATE","ACTUAL_DEMAND","YEAR","MONTH","WEEK","FORECASTED_DEMAND_INF","FORECASTED_DEMAND_SUP"]]

    metrics_df = pd.DataFrame([{
        "PARTITION_ID": context.partition_id.split('/')[1] if '/' in context.partition_id else context.partition_id,
        "ITEM_NUMBER_SITE": item_id,
        "FORECAST_ROWS": int(len(results)),
    }])
    context.upload_to_stage(metrics_df, "metrics.parquet",
                            write_function=lambda pdf, path: pdf.to_parquet(path, index=False))
    context.upload_to_stage(results, "forecast.parquet",
                            write_function=lambda pdf, path: pdf.to_parquet(path, index=False))

    return results


class ForecastUDTF:
    """Class which is registered as a UDTF to run demand forecasting."""
 
    def end_partition(self, df):
        """End partition method which utilizes the train_partition_function."""
        result_df = train_partition(df)
        result_df = result_df.rename(columns={c+"_OUT" for c in ["ITEM_NUMBER_SITE","PERFORM_DATE","ACTUAL_DEMAND"]})
        yield result_df


def _sanitize_col(c):
    return _re.sub(r'[^A-Za-z0-9_]', '', c).upper()


def get_train_version() -> str:
    return datetime.utcnow().strftime("v%Y%m%d_%H%M")


def prepare_data(session: Session):
    """Join demand data with item selection, save as FORECAST_INPUT_WEEKLY."""
    connect_cfg = utils.get_connection_config()
    feat_cfg = utils.get_feature_config()

    demand_period_long = session.table(feat_cfg['name'])
    active_items = demand_period_long.group_by('ITEM_NUMBER_SITE').count()
    active_items = active_items.filter(F.col("COUNT") < 1470)

    item_selection_tot = session.table("ITEM_SELECTION_TO_FCST").rename({
        'PURCHASE_CYCLE_CATEGORY_LAST_2_YEARS': 'PURCHASE_CYCLE_CATEGORY',
    })
    item_selection_tot = item_selection_tot.rename({c: _sanitize_col(c) for c in item_selection_tot.columns})

    item_selection_w = item_selection_tot.filter(
        F.col('PURCHASE_CYCLE_CATEGORY').isin(['Weekly', 'Daily'])
    )
    item_selection_w = item_selection_w.join(
        active_items.select('ITEM_NUMBER_SITE'), on='ITEM_NUMBER_SITE', how='inner'
    )

    forecast_input = demand_period_long.join(
        item_selection_w.select("ITEM_NUMBER_SITE", "PURCHASE_CYCLE_CATEGORY"),
        on="ITEM_NUMBER_SITE",
        how='inner',
    )

    partition_count = forecast_input.select(F.col("ITEM_NUMBER_SITE")).distinct().count()
    row_count = forecast_input.count()
    print(f"Preparing data: {partition_count:,} partitions, {row_count:,} rows")

    forecast_input.write.mode("overwrite").save_as_table("FORECAST_INPUT_WEEKLY")

    return session.table("FORECAST_INPUT_WEEKLY")


def execute_training(session: Session, run_id: str, forecast_input: sp.DataFrame):
    """Register UDTF and run distributed forecast across all partitions on warehouse."""

    input_cols = [c for c in forecast_input.columns if c not in [TIME, GRAIN]]

    vect_udtf_input_dtypes = [
        T.PandasDataFrameType(
            [
                field.datatype
                for field in forecast_input.schema.fields
                if field.name in input_cols
            ]
        )
    ]

    udtf_name = f"FORECAST_UDTF_{run_id}"
    session.udtf.register(
        ForecastUDTF,
        name=udtf_name,
        input_types=vect_udtf_input_dtypes,
        output_schema=T.PandasDataFrameType(
            [T.StringType(), T.TimestampType(T.TimestampTimeZone.NTZ), T.LongType(),
             T.LongType(), T.LongType(), T.LongType(),
             T.LongType(), T.LongType()],
            ["ITEM_NUMBER_SITE_OUT", "PERFORM_DATE_OUT", "ACTUAL_DEMAND_OUT",
             "YEAR", "MONTH", "WEEK",
             "FORECASTED_DEMAND_INF", "FORECASTED_DEMAND_SUP"],
        ),
        packages=[
            "snowflake-snowpark-python",
            "pandas",
            "numpy",
            "lightgbm",
            "scikit-learn",
            "scipy",
            "statsmodels",
            "darts",
            "tsfresh",
            "optuna",
            "mapie",
            "scikit-lego",
        ],
        artifact_repository="snowflake.snowpark.pypi_shared_repository",
        replace=True,
        is_permanent=False,
    )

    results = forecast_input.select(
        GRAIN,
        TIME,
        *[F.col(c) for c in input_cols],
        F.call_table_function(udtf_name, *input_cols).over(
            partition_by=[GRAIN],
            order_by=TIME,
        ),
    )

    return results


def collect_forecasts(session: Session, results: sp.DataFrame):
    """Write UDTF forecast results into WEEKLY_FORECASTS table."""
    session.sql("""
        CREATE TABLE IF NOT EXISTS WEEKLY_FORECASTS (
            ITEM_NUMBER_SITE VARCHAR,
            PERFORM_DATE TIMESTAMP_NTZ,
            ACTUAL_DEMAND NUMBER,
            YEAR NUMBER,
            MONTH NUMBER,
            WEEK NUMBER,
            FORECASTED_DEMAND_INF NUMBER,
            FORECASTED_DEMAND_SUP NUMBER
        );
    """).collect()

    forecast_cols = results.select(
        "ITEM_NUMBER_SITE", "PERFORM_DATE", "ACTUAL_DEMAND",
        "YEAR", "MONTH", "WEEK",
        "FORECASTED_DEMAND_INF", "FORECASTED_DEMAND_SUP",
    )
    forecast_cols.write.mode("overwrite").save_as_table("WEEKLY_FORECASTS")

    row_count = session.table("WEEKLY_FORECASTS").count()
    print(f"Collected {row_count:,} forecast rows into WEEKLY_FORECASTS")


def run_forecasting(session: Session = None) -> str:
    """Entry point: prepare data, run distributed forecast via UDTF, collect results."""
    train_version = get_train_version()
    run_id = f"forecast_{train_version}"

    if session is None:
        session = create_session(run_id)
        print(f"Connected: {session.get_current_account()}")
    else:
        session.sql(f"ALTER SESSION SET QUERY_TAG = '{run_id}'").collect()

    forecast_input = prepare_data(session)
    results = execute_training(session, run_id, forecast_input)
    collect_forecasts(session, results)

    return run_id


if __name__ == "__main__":
    session = create_session()
    print(f"Connected: {session.get_current_account()}")
    run_id = run_forecasting(session)
    print(f"RUN_ID={run_id}")
