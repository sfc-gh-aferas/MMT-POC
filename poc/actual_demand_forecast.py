
import pandas as pd
import numpy as np
import os
import random
import io
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import utils

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
#from weekly_features_for_daily_forecasts import features_w_14_all, features_prev_M_all, features_prev_6M_all

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

# imports not used
# import xlsxwriter
# from openpyxl import load_workbook

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


import optuna
from optuna.pruners import HyperbandPruner
import logging

ENABLE_OPTUNA_PRINTS = False   # Set to True to see optimization progress
ENABLE_LGBM_PRINTS = False    # Set to True to see LightGBM training progress

if not ENABLE_OPTUNA_PRINTS:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


from optuna.samplers import TPESampler
#from optuna.integration import PyTorchLightningPruningCallback # not used

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
        
        group['Year'] = group[date].dt.year
        group['Month'] = group[date].dt.month
        monthly_demand = group.groupby(['Year', 'Month'])['ACTUAL_DEMAND'].sum()
        num_months_with_demand = (monthly_demand > 0).sum()
        total_months = len(monthly_demand)

        results.append({
            'ITEM_NUMBER_SITE': item,
            'MEAN_PURCHASE_INTERVAL_(DAYS)': mean_interval,
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
            'MEAN_PURCHASE_INTERVAL_(DAYS)': mean_interval,
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
    device_type="gpu",
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
    df_sorted = df.sort_values(['Year', 'Month'])
    unique_months = df_sorted[['Year', 'Month']].drop_duplicates().values
    if len(unique_months) < 2:
        return 0  # Not enough data for previous month
    prev_year, prev_month = unique_months[-2]
    prev_month_rows = df[(df['Year'] == prev_year) & (df['Month'] == prev_month)]
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
        'Forecasted Demand Inf': [agg_inf],
        'Forecasted Demand Sup': [agg_sup]
    })


def simulate_aggregate_intervals(
    forecast_data,
    value_col='ACTUAL_DEMAND', 
    inf_col='Forecasted Demand Inf', 
    sup_col='Forecasted Demand Sup',             
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
        'Forecasted Demand Inf': [agg_inf],
        'Forecasted Demand Sup': [agg_sup]
    })

import re

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

def exclude_anomalous_last_year(df, group_col='ITEM_NUMBER_SITE', year_col='Year', days_col='EffectiveNbDays', threshold=0.3):
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
    pos_indices = forecast[(forecast['Month'] == month) & (forecast[demand_col] > 0)].index
    n_pos = len(pos_indices)
    
    # Use business logic demand_days for realistic distribution
    optimal_days = max(min(demand_days, 10), 1)  # Cap at 10, min 1
    possible_days = forecast[(forecast['Month'] == month)].index
    
    if diff > 0:  # Need to increase
        if n_pos >= optimal_days:
            # Use existing positive days if we have enough
            selected_indices = pos_indices[:optimal_days]
            add_per_day = int(np.ceil(diff / len(selected_indices)))
            forecast.loc[selected_indices, demand_col] += add_per_day
            forecast.loc[selected_indices, 'Forecasted Demand Inf'] += add_per_day
            forecast.loc[selected_indices, 'Forecasted Demand Sup'] += add_per_day
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
                forecast.at[idx, 'Forecasted Demand Inf'] = int(add_per_day * 0.95)
                forecast.at[idx, 'Forecasted Demand Sup'] = int(add_per_day * 1.15)
            print(f"Distributed {target_monthly_demand:.0f} units across {len(selected_indices)} days (business logic: {demand_days} days)")
        else:
            # Fallback: use whatever positive days we have or create minimal distribution
            selected_indices = pos_indices if n_pos > 0 else possible_days[:1]
            if len(selected_indices) > 0:
                add_per_day = int(np.ceil(target_monthly_demand / len(selected_indices)))
                for idx in selected_indices:
                    forecast.at[idx, demand_col] = add_per_day
                    forecast.at[idx, 'Forecasted Demand Inf'] = int(add_per_day * 0.95)
                    forecast.at[idx, 'Forecasted Demand Sup'] = int(add_per_day * 1.15)
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


# load from snowflake - One item for testing: 003259-2_102
# demand = pd.read_csv(fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\demand_sept_2025.csv', encoding='UTF-8', sep="|", decimal='.', doublequote=True, engine='python')
connect_cfg = utils.get_connection_config()
feat_cfg = utils.get_feature_config()

session = Session.builder.getOrCreate()
session.use_database(connect_cfg['database'])
session.use_schema(connect_cfg['schema'])
session.use_warehouse(connect_cfg['warehouse'])

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

demand['Year'] = demand['Perform Date'].dt.year
demand.drop(columns={'Year'}, inplace=True)
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
monthly_tot_all_items['Year'] = monthly_tot_all_items['Perform Date'].dt.year
monthly_tot_all_items['Month'] = monthly_tot_all_items['Perform Date'].dt.month
monthly_tot_all_items['DaysInMonth'] = monthly_tot_all_items['Perform Date'].dt.days_in_month
monthly_tot_all_items['EffectiveNbDays'] = 1
monthly_tot_all_items['EffectiveDemand'] = np.where(monthly_tot_all_items[demand_col] > 0, 1, 0)
monthly_tot_all_items = monthly_tot_all_items.groupby(['Item Number Site','Year','Month']).agg({'DaysInMonth':'max',                                                                 
                                                          'EffectiveNbDays': 'sum',
                                                          'EffectiveDemand': 'sum',
                                                          demand_col: 'sum'}).reset_index()

def is_incomplete(row):
    if row['Month'] == 2:
        return row['EffectiveNbDays'] < 28
    else:
        return row['EffectiveNbDays'] < 30

monthly_tot_all_items = monthly_tot_all_items.sort_values(['Item Number Site', 'Year', 'Month'])
mask = monthly_tot_all_items.groupby('Item Number Site').tail(1).apply(is_incomplete, axis=1)
last_indices = monthly_tot_all_items.groupby('Item Number Site').tail(1).index
monthly_tot_all_items = monthly_tot_all_items.drop(index=last_indices[mask])

years_per_item = monthly_tot_all_items.groupby('Item Number Site')['Year'].nunique().reset_index()

years_per_item.rename(columns={'Year': 'NumYears'}, inplace=True)
years_per_item = years_per_item[years_per_item['NumYears']>3].copy()
purchase_repurchase_cycle = pd.merge(purchase_repurchase_cycle, years_per_item[['Item Number Site']], on=['Item Number Site'], how='inner', sort = False)

purchase_repurchase_cycle2 = pd.merge(purchase_repurchase_cycle2, purchase_repurchase_cycle[['Item Number Site','Mean Purchase Interval (days)']], on=['Item Number Site'], how='left', sort=False)
purchase_repurchase_cycle2 = purchase_repurchase_cycle2[pd.isna(purchase_repurchase_cycle2['Mean Purchase Interval (days)_y'])].copy()
purchase_repurchase_cycle2.rename(columns={'Purchase Cycle Category2': 'Purchase Cycle Category'}, inplace=True)

years_per_item = monthly_tot_all_items.groupby('Item Number Site')['Year'].nunique().reset_index()
years_per_item.rename(columns={'Year': 'NumYears'}, inplace=True)
years_per_item = years_per_item[years_per_item['NumYears']>1].copy()
purchase_repurchase_cycle2 = pd.merge(purchase_repurchase_cycle2, years_per_item[['Item Number Site']], on=['Item Number Site'], how='inner', sort = False)

item_selection_bi_annual = purchase_repurchase_cycle2[purchase_repurchase_cycle2['Purchase Cycle Category']=='Bi-annual']
item_selection = purchase_repurchase_cycle[purchase_repurchase_cycle['Purchase Cycle Category'].isin(['Monthly', 'Weekly'])]
zz = demand[demand['Item Number Site']=='000006-000C_102_HENDTKUS']

demo = pd.merge(demand_period_long, item_selection[['Item Number Site']], on=['Item Number Site'], how='inner', sort=False)
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\demo.csv'
demo.to_csv(export_path, index=None, doublequote=False, header=True, sep=",", encoding='UTF-8')

def reduce_mem_usage(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        # Check if it fits in int32 to save even more
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')
    return df

zz1 = purchase_repurchase_cycle[purchase_repurchase_cycle['Item Number Site']=='047691-000_101_HENDTKUS']
zz2 = purchase_repurchase_cycle2[purchase_repurchase_cycle2['Item Number Site']=='047691-000_101_HENDTKUS']

item_selection = item_selection[item_selection['Item Number Site'].isin(['HF100259_400_HENDUK','41CXZ0018P_400_HENDUK','62001-640_400_HENDUK','15CXC0031P_400_HENDUK','10ARC0018_400_HENDUK'])]
item_selection_bi_annual = item_selection_bi_annual[item_selection_bi_annual['Item Number Site'].isin(['HS507425_400_HENDUK','30AXC0404_400_HENDUK','HS508656_400_HENDUK','11CLC0410_400_HENDUK','Y787110/LH_400_HENDUK'])]

item_selection = purchase_repurchase_cycle[purchase_repurchase_cycle['Item Number Site'].isin(['047691-000_101_HENDTKUS'])]
"""
demand_period_long = session.table(feat_cfg['name']).to_pandas()
demand_period_long['PERFORM_DATE'] = pd.to_datetime(demand_period_long['PERFORM_DATE'], format=ts_format)
demand_period_long['ITEM_NUMBER_SITE'] = demand_period_long['ITEM_NUMBER_SITE']
active_items = demand_period_long['ITEM_NUMBER_SITE'].value_counts().reset_index()

item_selection_tot = session.table("ITEM_SELECTION_TO_FCST").to_pandas()
item_selection_tot.rename(columns={'PURCHASE_CYCLE_CATEGORY_LAST_2_YEARS': 'PURCHASE_CYCLE_CATEGORY'}, inplace=True)
item_selection_m = item_selection_tot[item_selection_tot['PURCHASE_CYCLE_CATEGORY'].isin(['Monthly'])]
item_selection_m = pd.merge(item_selection_m, active_items[['ITEM_NUMBER_SITE']], on=['ITEM_NUMBER_SITE'], how='inner', sort=False)
#item_selection = item_selection_m.iloc[30:310].copy()

#Don't have these
#done_1 = pd.read_csv(fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\forecasts\all_intervals_lot1.csv', encoding='UTF-8', sep=";", decimal='.', doublequote=False, engine='python')
#done_2 = pd.read_csv(fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\forecasts\all_intervals_lot2.csv', encoding='UTF-8', sep=";", decimal='.', doublequote=False, engine='python')
#done = pd.concat([done_1, done_2], axis=0)
#done['done'] = 1
#item_selection_m = pd.merge(item_selection_m, done[['ITEM_NUMBER_SITE', 'done']], on=['ITEM_NUMBER_SITE'], how='left', sort=False)
#item_selection_m = item_selection_m[pd.isna(item_selection_m['done'])].copy()
item_selection_m.sort_values(['MEAN_PURCHASE_INTERVAL_(DAYS)'], ascending=[True], inplace=True)
item_selection = item_selection_m.head(20).copy()
zz = item_selection_m[item_selection_m['ITEM_NUMBER_SITE']=='059310-065_101_HENDTKUS']
#item_selection = item_selection_m[item_selection_m['ITEM_NUMBER_SITE']=="007500-491_101_HENDTKUS"].copy()

item_selection_w = item_selection_tot[item_selection_tot['PURCHASE_CYCLE_CATEGORY'].isin(['Weekly'])]
item_selection_w = pd.merge(item_selection_w, active_items[['ITEM_NUMBER_SITE']], on=['ITEM_NUMBER_SITE'], how='inner', sort=False)
item_selection_w.sort_values(['MEAN_PURCHASE_INTERVAL_(DAYS)'], ascending=[True], inplace=True)

item_selection = item_selection_w.head(8).copy()

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
for item_id in item_selection['ITEM_NUMBER_SITE']:
    print(item_id)
    item_cnt = item_cnt+1
    actual_category = item_selection.loc[item_selection['ITEM_NUMBER_SITE'] == item_id, 'PURCHASE_CYCLE_CATEGORY'].iloc[0]
    print(f"Processing item {item_id} with category: {actual_category}")
    print(f"This forecast represents {(item_cnt/item_selection.shape[0])*100:.1f}% of all forecasts to be done")
    
    # Initialize variables to avoid NameError issues
    monthly_forecast_max = 0  # Will be properly set in monthly processing, default to 0 for weekly items
    forecast_data = None
    
    demand_item_fct = demand_period_long[demand_period_long['ITEM_NUMBER_SITE'] == item_id].copy()  
    demand_item_fct.reset_index(drop=True, inplace=True)
    demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
    demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month
    demand_item_fct['DaysInMonth'] = demand_item_fct['PERFORM_DATE'].dt.days_in_month
    demand_item_fct['EffectiveNbDays'] = 1
    demand_item_fct['EffectiveDemand'] = np.where(demand_item_fct[demand_col] > 0, 1, 0)
    monthly_tot = demand_item_fct.groupby(['Year','Month']).agg({'DaysInMonth':'max',                                                                 
                                                              'EffectiveNbDays': 'sum',
                                                              'EffectiveDemand': 'sum',
                                                              demand_col: 'sum'}).reset_index()
    if (
    (monthly_tot.iloc[-1]['EffectiveNbDays'] < 28 and monthly_tot.iloc[-1]['Month'] == 2) or
    (monthly_tot.iloc[-1]['EffectiveNbDays'] < 30 and monthly_tot.iloc[-1]['Month'] != 2)
    ):
        monthly_tot = monthly_tot.iloc[:-1]
        
    demand_item_fct = pd.merge(demand_item_fct, monthly_tot[['Year','Month']], on=['Year','Month'], how='inner', sort=False)
    
    if item_selection.loc[item_selection['ITEM_NUMBER_SITE'] == item_id,
                          'PURCHASE_CYCLE_CATEGORY'].iloc[0] in ["Daily", "Weekly"]:
        if (
        (monthly_tot.iloc[0]['EffectiveNbDays'] < 28 and monthly_tot.iloc[0]['Month'] == 2) or
        (monthly_tot.iloc[0]['EffectiveNbDays'] < 30 and monthly_tot.iloc[0]['Month'] != 2)
        ):
            monthly_tot = monthly_tot.iloc[1:]
            
        demand_item_fct = pd.merge(demand_item_fct, monthly_tot[['Year','Month']], on=['Year','Month'], how='inner', sort=False)
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
        demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
        demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month        
        existing_days = demand_item_fct[
            (demand_item_fct['Year'] == last_year) & (demand_item_fct['Month'] == last_month)
        ]['PERFORM_DATE']
        missing_days = set(all_days) - set(existing_days)
        
        if missing_days:
            missing_rows = [{'ITEM_NUMBER_SITE': item_id, 'PERFORM_DATE': date} for date in sorted(missing_days)]
            missing_df = pd.DataFrame(missing_rows)
            demand_item_fct = pd.concat([demand_item_fct, missing_df], ignore_index=True, sort=False)
            demand_item_fct.sort_values('PERFORM_DATE', inplace=True)
            demand_item_fct.reset_index(drop=True, inplace=True)
            demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
            demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month        

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
        if item_selection.loc[item_selection['ITEM_NUMBER_SITE'] == item_id, 'PURCHASE_CYCLE_CATEGORY'].iloc[0] in ["Daily"]:
            days['DayOfWeek'] = days['PERFORM_DATE'].dt.dayofweek
            X_sup = pd.merge(X_sup, days[['DayOfWeek']], left_index=True, right_index=True)    
        X_sup = pd.merge(X_sup, days[['DayOfMonth_sin', 'DayOfMonth_cos']], left_index=True, right_index=True)
        
        demand_item_fct['MonthPeriod'] = demand_item_fct['PERFORM_DATE'].dt.to_period('M')
        demand_item_fct['FirstDayOfMonth'] = demand_item_fct['MonthPeriod'].dt.start_time
        demand_item_fct['Week'] = ((demand_item_fct['PERFORM_DATE'] - demand_item_fct['FirstDayOfMonth']).dt.days // 7) + 1
        demand_item_fct['Week'] = demand_item_fct['Week'].astype('Int64')
        demand_item_fct.drop(['MonthPeriod', 'FirstDayOfMonth'], axis=1, inplace=True)
        demand_item_fct['EffectiveNbDays'] = 1
        
        first_future_date = demand_item_fct['PERFORM_DATE'].max() - pd.DateOffset(months=6) + pd.Timedelta(days=1)
        first_future_month = first_future_date.to_period('M')        
        mask_future = demand_item_fct['PERFORM_DATE'] > last_date
        
        demand_item_fct.loc[mask_future, 'Week2'] = (
            (demand_item_fct.loc[mask_future, 'PERFORM_DATE'].dt.day - 1) // 7 + 1
        )
        
        demand_item_fct['Week2'] = demand_item_fct['Week2'].astype('Int64')
        
        future_months = (demand_item_fct.loc[mask_future, ['Year', 'Month']].drop_duplicates()
                         .sort_values(['Year', 'Month'])
                         .reset_index(drop=True))        
        next_weeks = {}
        for _, row in future_months.iterrows():
            year, month = row['Year'], row['Month']
            weeks = demand_item_fct[
                (demand_item_fct['Year'] == year) & (demand_item_fct['Month'] == month) & mask_future
            ]['Week2'].unique()
            next_weeks[(year, month)] = sorted(weeks)
        
        
        forecast_data = demand_item_fct.copy()
        forecast_data = forecast_data[~pd.isna(forecast_data[demand_col])]
        for col in ['Forecasted Demand Inf', 'Forecasted Demand Sup']:
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
        
        weekly_tot = demand_item_fct.groupby(['Year','Month','Week2']).agg({'EffectiveNbDays': 'sum', #'Day Name':'first',                                                                 
                                                                  'PERFORM_DATE': ['min', 'max']}).reset_index()
        weekly_tot.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in weekly_tot.columns]
        weekly_tot.rename(columns={'PERFORM_DATE_min': 'PERFORM_DATE', 'EffectiveNbDays_sum': 'EffectiveNbDays'}, inplace=True)
        weekly_tot = weekly_tot.sort_values('PERFORM_DATE')

        weekly_kpis = forecast_data.groupby(['ITEM_NUMBER_SITE', 'Year','Month','Week']).agg({demand_col: 'sum'}).reset_index()
        
        weekly_forecast=[]
        weekly_best_params = []
        total_week_idx = 0
        for (year, month), weeks in next_weeks.items():
            for week_idx, week in enumerate(weeks):
                total_week_idx += 1
                print(f"Year: {year}, Month: {month}, Week: {week}")
                train_item = forecast_data.groupby(['Year', 'Month', 'Week']).agg({
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
                    (weekly_tot['Year'] == year) &
                    (weekly_tot['Month'] == month) &
                    (weekly_tot['Week2'] == week)
                ]
                if not week_row.empty:
                    simulation_period_days = week_row.iloc[0]['EffectiveNbDays']
                else:
                    simulation_period_days = 7 
                simulation_period_days_prev = monthly_tot.iloc[-1]['DaysInMonth'] 
                
                demand_days = min(demand_days, simulation_period_days)
                weekly_forecast['Forecasted Effective Demand'] = [demand_days]                

                if week==1:
                    train_item_month = train_item.groupby(['Year', 'Month']).agg({
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
                demand_X.drop(columns=['Year','Month','DaysInMonth','EffectiveNbDays'], inplace=True, errors='ignore')            
                
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
                
                additional_feat = additional_feat.interpolate(method='spline', order=2, axis=0)
                additional_feat = additional_feat.sort_index()
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
                successful_forecast = False
                for dim_x in range(initial_dim_x, 0, -1):  # Try from initial_dim_x down to 1
                    try:
                        temp_preds = list()
                        for i, series in enumerate(feat_sc_next_days):
                            model = KalmanForecaster(dim_x=dim_x)
                            model.fit(series=series)
                            p = model.predict(simulation_period_days).pd_dataframe().reset_index()
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
                demand_X = reduce_mem_usage(demand_X)
                
                y_train = forecast_data[['PERFORM_DATE', demand_col]].copy()
                y_train.set_index('PERFORM_DATE', inplace=True)
                
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
                n_trials=30, callbacks=[print_callback])
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
                best_params_df['Week'] = week
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
                ci_05 = pd.DataFrame(ci_05, columns=['Forecasted Demand Inf'])
                pred = pd.DataFrame(pred, columns=[demand_col])
                ci_95 = pd.DataFrame(ci_95, columns=['Forecasted Demand Sup'])
                        
                forecast = pd.concat([ci_05, pred, ci_95], axis=1)
                for col in forecast.columns:
                    forecast[col] = forecast[col].abs().round(0)
                forecast['Forecasted Demand Sup'] = np.where(forecast[demand_col]==0,0,forecast['Forecasted Demand Sup'])
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
                    anchor_inf = forecast.loc[idx, 'Forecasted Demand Inf']
                    anchor_sup = forecast.loc[idx, 'Forecasted Demand Sup']
                    if idx > 0 and idx-1 not in selected:
                        anchor_val += forecast.loc[idx-1, demand_col]
                        anchor_inf += forecast.loc[idx-1, 'Forecasted Demand Inf']
                        anchor_sup += forecast.loc[idx-1, 'Forecasted Demand Sup']
                        new_values[idx-1] = 0
                        new_inf[idx-1] = 0
                        new_sup[idx-1] = 0
                    new_values[idx] = anchor_val
                    new_inf[idx] = anchor_inf
                    new_sup[idx] = anchor_sup
            
                forecast[demand_col] = new_values
                forecast['Forecasted Demand Inf'] = new_inf
                forecast['Forecasted Demand Sup'] = new_sup                
                
                forecast['EffectiveDemand'] = np.where(forecast[demand_col]>0,1,0)
                forecast['Year'] = forecast['PERFORM_DATE'].dt.year
                forecast['Month'] = forecast['PERFORM_DATE'].dt.month
                forecast['Week'] = week
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
                forecast_data['Forecasted Demand Inf'] = forecast_data['Forecasted Demand Inf'].round(0).astype('Int64')
                forecast_data['Forecasted Demand Sup'] = forecast_data['Forecasted Demand Sup'].round(0).astype('Int64')
                
                if week == 5:
                    week_sums = forecast_data[
                        (forecast_data['Year'] == year) &
                        (forecast_data['Month'] == month)
                    ].groupby('Week')[demand_col].sum()
                    max_weekly_value = week_sums.max()
                    max_week = week_sums.idxmax()
                    if max_weekly_value < monthly_forecast_max:
                        week_mask = (
                            (forecast_data['Year'] == year) &
                            (forecast_data['Month'] == month) &
                            (forecast_data['Week'] == max_week)
                        )
                        week_days = forecast_data[week_mask].index.tolist()

                        week_values = forecast_data.loc[week_days, demand_col]
                        max_day_value = week_values.max()
                        max_day_indices = week_values[week_values == max_day_value].index.tolist()

                        np.random.seed(RANDOM_SEED)
                        chosen_idx = np.random.choice(max_day_indices)
                        diff = monthly_forecast_max - max_weekly_value

                        forecast_data.at[chosen_idx, demand_col] += diff
                        forecast_data.at[chosen_idx, 'Forecasted Demand Inf'] += diff
                        forecast_data.at[chosen_idx, 'Forecasted Demand Sup'] += diff
        
        forecast_data['Forecasted Demand Inf'] = np.minimum(
            forecast_data['Forecasted Demand Inf'],
            forecast_data['Forecasted Demand Sup']
        )
        forecast_data['Forecasted Demand Sup'] = np.maximum.reduce([
            forecast_data['Forecasted Demand Sup'],
            forecast_data['Forecasted Demand Inf'],
            forecast_data[demand_col]
        ])
        forecast_data['EffectiveDemand'] = np.where(forecast_data['ACTUAL_DEMAND'] > 0,1,0)
        forecast_data['Forecasted Demand Inf'] = forecast_data['Forecasted Demand Inf'].astype(float)
        forecast_data['Forecasted Demand Sup'] = forecast_data['Forecasted Demand Sup'].astype(float)
        mask = (forecast_data['ACTUAL_DEMAND'] > 0) & (forecast_data['ACTUAL_DEMAND'] < forecast_data['Forecasted Demand Inf'])
        forecast_data.loc[mask, 'Forecasted Demand Inf'] = forecast_data.loc[mask, 'ACTUAL_DEMAND'] * 0.95
        forecast_data.loc[mask, 'Forecasted Demand Sup'] = forecast_data.loc[mask, 'ACTUAL_DEMAND'] * 1.15
        forecast_data['Forecasted Demand Inf'] = forecast_data['Forecasted Demand Inf'].round(0)
        forecast_data['Forecasted Demand Sup'] = forecast_data['Forecasted Demand Sup'].round(0)
        if np.issubdtype(forecast_data['Forecasted Demand Inf'].dtype, np.floating):
            forecast_data['Forecasted Demand Inf'] = forecast_data['Forecasted Demand Inf'].astype('Int64')
        if np.issubdtype(forecast_data['Forecasted Demand Sup'].dtype, np.floating):
            forecast_data['Forecasted Demand Sup'] = forecast_data['Forecasted Demand Sup'].astype('Int64')
        
        # Apply confidence multiplier if structural breaks were detected
        # Use the variables defined earlier instead of .attrs to avoid tsfresh conflicts
        confidence_multiplier = getattr(forecast_data, 'attrs', {}).get('confidence_multiplier', 1.0)
        break_info = getattr(forecast_data, 'attrs', {}).get('break_info', None)
        
        if confidence_multiplier_result > 1.0:
            
            # Expand confidence intervals around the central forecast
            central_forecast = forecast_data[demand_col]
            lower_diff = central_forecast - forecast_data['Forecasted Demand Inf']
            upper_diff = forecast_data['Forecasted Demand Sup'] - central_forecast
            
            # Apply multiplier to the differences (expanding uncertainty)
            forecast_data['Forecasted Demand Inf'] = central_forecast - (lower_diff * confidence_multiplier_result)
            forecast_data['Forecasted Demand Sup'] = central_forecast + (upper_diff * confidence_multiplier_result)
            
            # for non-negative intervals
            forecast_data['Forecasted Demand Inf'] = np.maximum(forecast_data['Forecasted Demand Inf'], 0)
            
        #all_best_params.append(pd.concat(weekly_best_params, ignore_index=True))

        true_forecast = forecast_data[forecast_data['PERFORM_DATE'] > last_date_hist].copy()
        global_interval = simulate_aggregate_intervals(true_forecast, dist='nbinom')
        global_interval['ITEM_NUMBER_SITE'] = item_id
        global_interval['Ratio1'] = global_interval['Forecasted Demand Inf'] / global_interval[demand_col]
        global_interval['Ratio2'] = global_interval['Forecasted Demand Sup'] / global_interval[demand_col]
        last_date = forecast_data['PERFORM_DATE'].max()
        start_date = last_date - pd.DateOffset(months=horizon)
        yearly_tot = forecast_data.loc[
            (forecast_data['PERFORM_DATE'] > start_date) & (forecast_data['PERFORM_DATE'] <= last_date),
            demand_col
        ].sum()
        last_year = forecast_data.loc[
            (forecast_data['PERFORM_DATE'] > start_date) & (forecast_data['PERFORM_DATE'] <= last_date)
        ]

        global_interval['Forecasted Demand Inf'] = global_interval['Ratio1'] * yearly_tot
        global_interval['Forecasted Demand Sup'] = global_interval['Ratio2'] * yearly_tot
        
        # Handle NaN and infinite values before converting to int
        global_interval['Forecasted Demand Inf'] = global_interval['Forecasted Demand Inf'].abs()
        global_interval['Forecasted Demand Sup'] = global_interval['Forecasted Demand Sup'].abs()
        
        # Replace NaN and infinite values with 0 before converting to int
        global_interval['Forecasted Demand Inf'] = global_interval['Forecasted Demand Inf'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        global_interval['Forecasted Demand Sup'] = global_interval['Forecasted Demand Sup'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        global_interval['Forecasted Demand'] = yearly_tot
        global_interval = global_interval[['ITEM_NUMBER_SITE', 'Forecasted Demand', 'Forecasted Demand Inf', 'Forecasted Demand Sup']]

        forecast_results_summary = forecast_data.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month', 'Week']).agg(
            Nb_Days_sum=('EffectiveDemand', 'sum'),
            Forecasted_Demand_sum=(demand_col, 'sum'),
            Forecasted_Demand_mean=(demand_col, 'mean'),
            Forecasted_Demand_std=(demand_col, 'std'),
            Forecasted_Demand_inf_sum=('Forecasted Demand Inf', 'sum'),
            Forecasted_Demand_inf_mean=('Forecasted Demand Inf', 'mean'),
            Forecasted_Demand_inf_std=('Forecasted Demand Inf', 'std'),
            Forecasted_Demand_sup_sum=('Forecasted Demand Sup', 'sum'),
            Forecasted_Demand_sup_mean=('Forecasted Demand Sup', 'mean'),
            Forecasted_Demand_sup_std=('Forecasted Demand Sup', 'std'),
        ).reset_index()

        forecast_results_summary.rename(columns={
            'Nb_Days_sum': 'Number of Days of Demand',
            'Forecasted_Demand_sum': 'Forecasted Demand',
            'Forecasted_Demand_mean': 'Forecasted Demand mean',
            'Forecasted_Demand_std': 'Forecasted Demand std',
            'Forecasted_Demand_inf_sum': 'Forecasted Demand Inf',
            'Forecasted_Demand_inf_mean': 'Forecasted Demand Inf mean',
            'Forecasted_Demand_inf_std': 'Forecasted Demand Inf std',
            'Forecasted_Demand_sup_sum': 'Forecasted Demand Sup',
            'Forecasted_Demand_sup_mean': 'Forecasted Demand Sup mean',
            'Forecasted_Demand_sup_std': 'Forecasted Demand Sup std',
        }, inplace=True)

        forecast_results_summary['Forecasted Demand mean'] = forecast_results_summary['Forecasted Demand mean'].round(3)
        forecast_results_summary['Forecasted Demand std'] = forecast_results_summary['Forecasted Demand std'].round(3)
        forecast_results_summary['Forecasted Demand Inf mean'] = forecast_results_summary['Forecasted Demand Inf mean'].round(3)
        forecast_results_summary['Forecasted Demand Inf std'] = forecast_results_summary['Forecasted Demand Inf std'].round(3)
        forecast_results_summary['Forecasted Demand Sup mean'] = forecast_results_summary['Forecasted Demand Sup mean'].round(3)
        forecast_results_summary['Forecasted Demand Sup std'] = forecast_results_summary['Forecasted Demand Sup std'].round(3)

    if item_selection.loc[item_selection['ITEM_NUMBER_SITE'] == item_id,
                          'PURCHASE_CYCLE_CATEGORY'].iloc[0] in ["Daily", "Weekly"]:
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


###############  Monthly
        
    if item_selection.loc[item_selection['ITEM_NUMBER_SITE'] == item_id, 
                          'PURCHASE_CYCLE_CATEGORY'].iloc[0] in ["Monthly"]:
        
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
        demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
        demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month        
        existing_days = demand_item_fct[
            (demand_item_fct['Year'] == last_year) & (demand_item_fct['Month'] == last_month)
        ]['PERFORM_DATE']
        missing_days = set(all_days) - set(existing_days)
        
        if missing_days:
            missing_rows = [{'ITEM_NUMBER_SITE': item_id, 'PERFORM_DATE': date} for date in sorted(missing_days)]
            missing_df = pd.DataFrame(missing_rows)
            demand_item_fct = pd.concat([demand_item_fct, missing_df], ignore_index=True, sort=False)
            demand_item_fct.sort_values('PERFORM_DATE', inplace=True)
            demand_item_fct.reset_index(drop=True, inplace=True)
            demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
            demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month        

        #Wcut_off = pd.to_datetime('2025-07-31 00:00:00', format=ts_format_2)
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
        X_sup = pd.merge(X_sup, days[['DayOfMonth_sin', 'DayOfMonth_cos']], left_index=True, right_index=True)
        
        demand_item_fct['MonthPeriod'] = demand_item_fct['PERFORM_DATE'].dt.to_period('M')
        demand_item_fct['FirstDayOfMonth'] = demand_item_fct['MonthPeriod'].dt.start_time
        demand_item_fct['Week'] = ((demand_item_fct['PERFORM_DATE'] - demand_item_fct['FirstDayOfMonth']).dt.days // 7) + 1
        demand_item_fct['Week'] = demand_item_fct['Week'].astype('Int64')
        demand_item_fct.drop(['MonthPeriod', 'FirstDayOfMonth'], axis=1, inplace=True)
        demand_item_fct['EffectiveNbDays'] = 1
       
        first_future_date = demand_item_fct['PERFORM_DATE'].max() - pd.DateOffset(months=6) + pd.Timedelta(days=1)
        first_future_month = first_future_date.to_period('M')        
        mask_future = demand_item_fct['PERFORM_DATE'] > last_date
        
        demand_item_fct.loc[mask_future, 'Week2'] = (
            (demand_item_fct.loc[mask_future, 'PERFORM_DATE'].dt.day - 1) // 7 + 1
        )
        
        demand_item_fct['Week2'] = demand_item_fct['Week2'].astype('Int64')
        
        future_months = (demand_item_fct.loc[mask_future, ['Year', 'Month']].drop_duplicates()
                         .sort_values(['Year', 'Month'])
                         .reset_index(drop=True))
        next_months = {(row['Year'], row['Month']): [1] for _, row in future_months.iterrows()}
        
        forecast_data = demand_item_fct.copy()
        forecast_data = forecast_data[~pd.isna(forecast_data[demand_col])]
        for col in ['Forecasted Demand Inf', 'Forecasted Demand Sup']:
            if col not in forecast_data.columns:
                forecast_data[col] = np.nan                
        forecast_data['History'] = 1
        
        non_zero_months = (monthly_tot['ACTUAL_DEMAND'] > 0).sum()
        # Initial dim_x value based on data availability
        if non_zero_months < 6:
            initial_dim_x = 8
        else:
            initial_dim_x = 6

        
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
        
        monthly_tot = demand_item_fct.groupby(['Year','Month']).agg({'EffectiveNbDays': 'sum',                                             
                                                                  'PERFORM_DATE': ['min', 'max']}).reset_index()
        monthly_tot.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in monthly_tot.columns]
        monthly_tot.rename(columns={'PERFORM_DATE_min': 'PERFORM_DATE', 'EffectiveNbDays_sum': 'EffectiveNbDays'}, inplace=True)
        monthly_tot = monthly_tot.sort_values('PERFORM_DATE').reset_index(drop=True)

        monthly_tot_demand = forecast_data.groupby(['Year','Month'])[demand_col].sum().reset_index()                                             
                    
        monthly_forecast=[]
        monthly_best_params = []

        for idx, (year, month) in enumerate(next_months):
            train_item = forecast_data.groupby(['Year', 'Month']).agg({
                'EffectiveDemand': 'sum',
                'EffectiveNbDays': 'sum',
                'PERFORM_DATE': 'min',
                demand_col: 'sum'
            }).reset_index()
            train_item['Year'] = train_item['Year'].astype(int)
            train_item['Month'] = train_item['Month'].astype(int)
            train_item['Date'] = pd.to_datetime(train_item['Year'].astype(str) + '-' + train_item['Month'].astype(str).str.zfill(2) + '-01')
            train_item = train_item.sort_values('Date')
            y = train_item.set_index('Date')['EffectiveDemand'].asfreq('MS')
            
            alphas = [0.1, 0.2, 0.5, 0.6]
            betas = [0.1, 0.2, 0.5, 0.6]

            best_mae_es_d = np.inf
            best_params = None
            best_forecast = None
            split_idx = int(len(y) * 0.8)
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
                monthly_forecast = fit_full.forecast(1).values[0]
                monthly_forecast = pd.DataFrame({'Forecasted Effective Demand': [monthly_forecast]})
          
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
                    trend='add',
                    seasonal=None,
                    initialization_method="estimated"
                )
                fit_full = model_full.fit(
                    smoothing_level=max(best_params[0], min(alphas)),
                    smoothing_trend=max(best_params[1], min(betas)),
                    optimized=False
                )
                monthly_forecast = fit_full.forecast(1).values[0]
                monthly_forecast = pd.DataFrame({'Forecasted Effective Demand': [monthly_forecast]})

            demand_days_es = np.ceil(monthly_forecast['Forecasted Effective Demand'].iloc[0]).astype(int)

            y.reset_index(drop=True, inplace=True)
            ts_train_item = TimeSeries.from_dataframe(y.to_frame())
            
            base_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            if idx + 1 <= 3:
                lags_to_test = base_lags
            elif 4 <= idx + 1 <= 8:
                lags_to_test = base_lags + [10, 12]
            elif 9 <= idx + 1 <= 11:
                lags_to_test = base_lags + [10, 12, 18]
            else:
                lags_to_test = base_lags + [10, 12, 18, 24]            

            best_lag = None
            best_mae_lr_d = float('inf')           
            split_idx = int(len(ts_train_item) * 0.8)
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
            
            y = train_item.set_index('Date')[demand_col].asfreq('MS')            
            alphas = [0.2, 0.25, 0.35, 0.5, 0,55, 0.6]
            betas = [0.1, 0.2, 0.35, 0.5, 0,55, 0.6]
            
            best_mae_es = np.inf        
            if demand_days_es > 0:    
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
                    demand_monthly_forecast_es = fit_full.forecast(1).values[0]
               
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
                        trend='add',
                        seasonal=None,
                        initialization_method="estimated"
                    )
                    fit_full = model_full.fit(
                        smoothing_level=max(best_params[0], min(alphas)),
                        smoothing_trend=max(best_params[1], min(betas)),
                        optimized=False
                    )
                    demand_monthly_forecast_es = fit_full.forecast(1).values[0]
            else:
                demand_monthly_forecast_es = 0
            
            ts_train_item = TimeSeries.from_dataframe(y.to_frame())
            base_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            if idx + 1 <= 3:
                lags_to_test = base_lags
            elif 4 <= idx + 1 <= 8:
                lags_to_test = base_lags + [10, 12]
            elif 9 <= idx + 1 <= 11:
                lags_to_test = base_lags + [10, 12, 18]
            else:
                lags_to_test = base_lags + [10, 12, 18, 24]            
            best_lag = None
            best_mae_lr = float('inf')           
            split_idx = int(len(ts_train_item) * 0.7)
            train_ts, val_ts = ts_train_item[:split_idx], ts_train_item[split_idx:]
            
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
            demand_monthly_forecast_lr = model.predict(1)
            demand_monthly_forecast_lr = float(demand_monthly_forecast_lr.values()[0, 0])
            
            # Calculate historical zero frequency for this month
            same_month_historical = monthly_tot_demand[(monthly_tot_demand['Month'] == month) & (monthly_tot_demand['Year'] < year)]
            if not same_month_historical.empty:
                total_same_months = len(same_month_historical)
                zero_months = len(same_month_historical[same_month_historical[demand_col] == 0])
                zero_frequency = zero_months / total_same_months if total_same_months > 0 else 0
            else:
                zero_frequency = 0
                        
            demand_es_lr_ratio = 100
            
            if (best_mae_es > 0) and (best_mae_lr > 0):
                if demand_monthly_forecast_es > 0 and demand_monthly_forecast_lr > 0:
                    wgt1 = (1/best_mae_es) / (1/best_mae_es + 1/best_mae_lr)
                    wgt2 = (1/best_mae_lr) / (1/best_mae_es + 1/best_mae_lr)
                    demand_monthly_forecast_w = int(np.round(wgt1 * demand_monthly_forecast_es + wgt2 * demand_monthly_forecast_lr))
                    demand_es_lr_ratio = (abs(demand_monthly_forecast_es - demand_monthly_forecast_lr)/max(demand_monthly_forecast_lr, 1))*100
                    
                elif demand_monthly_forecast_es == 0 and demand_monthly_forecast_lr > 0:
                    if zero_frequency > 0.3:  # Historically, this month often has zero demand
                        # ES zero prediction is credible - give it some weight
                        es_confidence = zero_frequency  # Higher zero frequency = more ES confidence
                        lr_confidence = 1 - zero_frequency
                        demand_monthly_forecast_w = int(np.round(lr_confidence * demand_monthly_forecast_lr))
                    else:
                        # ES zero prediction is suspicious - use mostly LR
                        demand_monthly_forecast_w = demand_monthly_forecast_lr
                        
                elif demand_monthly_forecast_es > 0 and demand_monthly_forecast_lr == 0:
                    if zero_frequency > 0.3:
                        demand_monthly_forecast_w = int(np.round(demand_monthly_forecast_es * (1 - zero_frequency)))
                    else:
                        demand_monthly_forecast_w = demand_monthly_forecast_es
                        
                else:
                    demand_monthly_forecast_w = 0
                    
            elif best_mae_lr > 0 and demand_monthly_forecast_lr > 0:
                demand_monthly_forecast_w = demand_monthly_forecast_lr
            elif best_mae_es > 0 and demand_monthly_forecast_es > 0:
                demand_monthly_forecast_w = demand_monthly_forecast_es
            else:
                demand_monthly_forecast_w = 0

            days_in_month = calendar.monthrange(year, month)[1]
            if demand_days > days_in_month:
                demand_days = days_in_month
            monthly_forecast['Forecasted Effective Demand'] = [demand_days]                
            
            mask_current = (monthly_tot['Year'] == year) & (monthly_tot['Month'] == month)
            simulation_period_days = int(monthly_tot.loc[mask_current, 'EffectiveNbDays'].values[0]) if mask_current.any() else None
            
            if idx > 0:
                prev_year, prev_month = list(next_months)[idx - 1]
                mask_prev = (monthly_tot['Year'] == prev_year) & (monthly_tot['Month'] == prev_month)
                simulation_period_days_prev = int(monthly_tot.loc[mask_prev, 'EffectiveNbDays'].values[0]) if mask_prev.any() else None
            else:
                simulation_period_days_prev = np.nan
            
            if np.isnan(simulation_period_days_prev):
                if simulation_period_days == 31 and month != 2:
                    simulation_period_days_prev = 30
                elif simulation_period_days == 30 and month != 2:
                    simulation_period_days_prev = 31
                else:
                    simulation_period_days_prev = 31
           
            num_days = calendar.monthrange(year, month)[1]
            month_start = pd.Timestamp(year=year, month=month, day=1)
            month_end = pd.Timestamp(year=year, month=month, day=num_days)
            next_days = pd.DataFrame({'PERFORM_DATE': pd.date_range(start=month_start, end=month_end, freq='D')})
            demand_y = forecast_data[['PERFORM_DATE', demand_col]].copy()
            demand_X = forecast_data.copy()
            demand_y = demand_y[~pd.isna(demand_y[demand_col])]
            demand_X = demand_X[~pd.isna(demand_X[demand_col])]
            
            df = demand_X[['PERFORM_DATE',demand_col]].copy()
            df['PERFORM_DATE'] = df['PERFORM_DATE'].apply(str)
            df['Duplic'] = demand_col
            Tri = 'PERFORM_DATE'
            nb_months = 1
            max_nb_months = 12
            while df[demand_col].iloc[-nb_months * simulation_period_days:].nunique() == 1 and nb_months < max_nb_months:
                print(f"Validation period contains only identical values. Expanding to {nb_months + 1} months...")
                nb_months += 1
            # Convert feature lists to tsfresh dictionary format
            features_curr_M_list = [
            'ACTUAL_DEMAND__index_mass_quantile__q_0.9'                                                             ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2"'                          ,
            'ACTUAL_DEMAND__energy_ratio_by_chunks__num_segments_10__segment_focus_9'                               ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4"'                          ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6"'                          ,
            'ACTUAL_DEMAND__last_location_of_minimum'                                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__index_mass_quantile__q_0.8'                                                             ,
            'ACTUAL_DEMAND__agg_autocorrelation__f_agg_"var"__maxlag_40"'                                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_3"'                                                ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"'                        ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"'                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_9"'                                                ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"'                        ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_1"'                                                ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0"'                          ,
            'ACTUAL_DEMAND__mean_change'                                                                            ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"'                     ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_2"'                                                ,
            'ACTUAL_DEMAND__mean_second_derivative_central'                                                         ,
            'ACTUAL_DEMAND__index_mass_quantile__q_0.7'                                                             ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"'                      ,
            'ACTUAL_DEMAND__linear_trend__attr_"rvalue"'                                                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_4"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_7"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_6"'                                                ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"'                           ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_8"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_15"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_5"'                                                ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"var"'                           ,
            'ACTUAL_DEMAND__index_mass_quantile__q_0.6'                                                             ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"'                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_14"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var"'                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_10"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_9"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"'                       ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"'                           ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"'                      ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"'                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_5"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_16"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"'                      ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_11"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"'                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_13"'                                               ,
            'ACTUAL_DEMAND__mean'                                                                                   ,
            'ACTUAL_DEMAND__mean_n_absolute_max__number_of_maxima_7'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_8"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_0"'                                                 ,
            'ACTUAL_DEMAND__sum_values'                                                                             ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_0"'                                                ,
            'ACTUAL_DEMAND__cid_ce__normalize_True'                                                                 ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_12"'                                               ,
            'ACTUAL_DEMAND__count_below__t_0'                                                                       ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"'                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_14"'                                              ,
            'ACTUAL_DEMAND__count_above_mean'                                                                       ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)'                                 ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_6"'                                               ,
            'ACTUAL_DEMAND__quantile__q_0.9'                                                                        ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_16"'                                              ,
            'ACTUAL_DEMAND__fft_aggregated__aggtype_"skew"'                                                       ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_13"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_12"'                                              ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_10__w_2__widths_(2, 5, 10, 20)'                                 ,
            'ACTUAL_DEMAND__binned_entropy__max_bins_10'                                                            ,
            'ACTUAL_DEMAND__last_location_of_maximum'                                                               ,
            'ACTUAL_DEMAND__root_mean_square'                                                                       ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_11__w_2__widths_(2, 5, 10, 20)'                                 ,
            'ACTUAL_DEMAND__abs_energy'                                                                             ,
            'ACTUAL_DEMAND__first_location_of_maximum'                                                              ,
            'ACTUAL_DEMAND__lempel_ziv_complexity__bins_100'                                                        ,
            'ACTUAL_DEMAND__ratio_beyond_r_sigma__r_1'                                                              ,
            'ACTUAL_DEMAND__agg_autocorrelation__f_agg_"mean"__maxlag_40"'                                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_13"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_9"'                                                 ,
            'ACTUAL_DEMAND__linear_trend__attr_"slope"'                                                           ,
            'ACTUAL_DEMAND__linear_trend__attr_"intercept"'                                                       ,
            'ACTUAL_DEMAND__lempel_ziv_complexity__bins_10'                                                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_9"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_12"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_4"'                                                ,
            'ACTUAL_DEMAND__ratio_beyond_r_sigma__r_1.5'                                                            ,
            'ACTUAL_DEMAND__variance'                                                                               ,
            'ACTUAL_DEMAND__index_mass_quantile__q_0.4'                                                             ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__ar_coefficient__coeff_0__k_10'                                                          ,
            'ACTUAL_DEMAND__fft_aggregated__aggtype_"kurtosis"'                                                   ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_8"'                                                ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)'                                ,
            'ACTUAL_DEMAND__standard_deviation'                                                                     ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_13__w_2__widths_(2, 5, 10, 20)'                                 ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_6__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_3"'                                                 ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__fft_aggregated__aggtype_"centroid"'                                                   ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"'                       ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"'                          ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)'                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_11"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_5"'                                                ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)'                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_4"'                                                 ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)'                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_11"'                                              ,
            'ACTUAL_DEMAND__symmetry_looking__r_0.15000000000000002'                                                ,
            'ACTUAL_DEMAND__agg_autocorrelation__f_agg_"median"__maxlag_40"'                                       ,
            'ACTUAL_DEMAND__number_peaks__n_1'                                                                      ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_14"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_16"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_15"'                                               ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__lempel_ziv_complexity__bins_3'                                                          ,
            'ACTUAL_DEMAND__ar_coefficient__coeff_3__k_10'                                                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_7"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_6"'                                                ,
            'ACTUAL_DEMAND__first_location_of_minimum'                                                              ,
            'ACTUAL_DEMAND__large_standard_deviation__r_0.30000000000000004'                                        ,
            'ACTUAL_DEMAND__ar_coefficient__coeff_1__k_10'                                                          ,
            'ACTUAL_DEMAND__ar_coefficient__coeff_4__k_10'                                                          ,
            'ACTUAL_DEMAND__lempel_ziv_complexity__bins_5'                                                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_8"'                                                 ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"'                        ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"var"'                         ,
            'ACTUAL_DEMAND__ratio_value_number_to_time_series_length'                                               ,
            'ACTUAL_DEMAND__lempel_ziv_complexity__bins_2'                                                          ,
            'ACTUAL_DEMAND__symmetry_looking__r_0.1'                                                                ,
            'ACTUAL_DEMAND__large_standard_deviation__r_0.25'                                                       ,
            'ACTUAL_DEMAND__ratio_beyond_r_sigma__r_2'                                                              ,
            'ACTUAL_DEMAND__number_cwt_peaks__n_5'                                                                  ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"'                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_27"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_24"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_23"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_22"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_19"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_18"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_17"'                                               ,
            'ACTUAL_DEMAND__energy_ratio_by_chunks__num_segments_10__segment_focus_6'                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_7"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_14"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_27"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_25"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_24"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_22"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_17"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_15"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_1"'                                                 ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"abs"__coeff_2"'                                                 ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_10"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_25"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_3"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_31"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_30"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_29"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_28"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_26"'                                               ,
            'ACTUAL_DEMAND__variation_coefficient'
            ]
            
            features_6_M_list = [
            'ACTUAL_DEMAND__last_location_of_minimum'                                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"'                      ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"'                       ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"'                          ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8"'                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_27"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"'                           ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"'                      ,
            'ACTUAL_DEMAND__energy_ratio_by_chunks__num_segments_10__segment_focus_9'                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_52"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"'                        ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_80"'                                              ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"'                         ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"'                          ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"'                     ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4"'                          ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2"'                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_27"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_53"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_80"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_54"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_26"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_80"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"'                        ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6"'                          ,
            'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0"'                          ,
            'ACTUAL_DEMAND__mean_change'                                                                            ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_54"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_28"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_56"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_55"'                                               ,
            'ACTUAL_DEMAND__mean_second_derivative_central'                                                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_51"'                                               ,
            'ACTUAL_DEMAND__index_mass_quantile__q_0.9'                                                             ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_52"'                                              ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_11__w_2__widths_(2, 5, 10, 20)'                                 ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_12__w_2__widths_(2, 5, 10, 20)'                                 ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"'                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_77"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_78"'                                              ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"'                         ,
            'ACTUAL_DEMAND__agg_autocorrelation__f_agg_"var"__maxlag_40"'                                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_24"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_72"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_48"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"'                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_71"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_48"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_47"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_53"'                                              ,
            'ACTUAL_DEMAND__fft_aggregated__aggtype_"skew"'                                                       ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_73"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_47"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_46"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"'                         ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_50"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_49"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_49"'                                              ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"'                          ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_45"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_73"'                                              ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"'                          ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_49"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"'                          ,
            'ACTUAL_DEMAND__index_mass_quantile__q_0.8'                                                             ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_98"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_29"'                                               ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_98"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_97"'                                              ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_97"'                                               ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)'                                  ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_95"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_94"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_6"'                                                ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_57"'                                               ,
            'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"mean"'                        ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_23"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_96"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_95"'                                               ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_94"'                                               ,
            'ACTUAL_DEMAND__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)'                                 ,
            'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_74"'            
                ]
            
            features_12_M_list = [
                'ACTUAL_DEMAND__last_location_of_minimum'                                                                    ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"'                             ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"'                            ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"'                               ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"'                          ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_53"'                                                    ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"'                           ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"'                              ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"'                              ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"'                               ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_52"'                                                    ,
                'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4"'                               ,
                'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8"'                               ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"'                              ,
                'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6"'                               ,
                'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2"'                               ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"'                               ,
                'ACTUAL_DEMAND__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)'                                       ,
                'ACTUAL_DEMAND__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)'                                       ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"'                                ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"'                           ,
                'ACTUAL_DEMAND__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)'                                       ,
                'ACTUAL_DEMAND__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)'                                       ,
                'ACTUAL_DEMAND__mean_change'                                                                                 ,
                'ACTUAL_DEMAND__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0"'                               ,
                'ACTUAL_DEMAND__fft_aggregated__aggtype_"skew"'                                                            ,
                'ACTUAL_DEMAND__cwt_coefficients__coeff_12__w_2__widths_(2, 5, 10, 20)'                                      ,
                'ACTUAL_DEMAND__mean_second_derivative_central'                                                              ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_54"'                                                    ,
                'ACTUAL_DEMAND__energy_ratio_by_chunks__num_segments_10__segment_focus_9'                                    ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"'                               ,
                'ACTUAL_DEMAND__agg_autocorrelation__f_agg_"var"__maxlag_40"'                                               ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_48"'                                                    ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"'                               ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_96"'                                                    ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"'                              ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_98"'                                                   ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_97"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_95"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_52"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_97"'                                                   ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_94"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_96"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_97"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"angle"__coeff_52"'                                                   ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_95"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_47"'                                                    ,
                'ACTUAL_DEMAND__first_location_of_minimum'                                                                   ,
                'ACTUAL_DEMAND__index_mass_quantile__q_0.9'                                                                  ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"'                              ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_98"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_94"'                                                    ,
                'ACTUAL_DEMAND__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"'                             ,
                'ACTUAL_DEMAND__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)'                                       ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"real"__coeff_49"'                                                    ,
                'ACTUAL_DEMAND__fft_coefficient__attr_"imag"__coeff_93"'            
                ]
            
            def parse_feature_name_to_tsfresh_config(feature_name):
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
                    func_name, params = parse_feature_name_to_tsfresh_config(feature_name)
                    if func_name:
                        if func_name not in config:
                            config[func_name] = [] if params else None
                        
                        if params and config[func_name] is not None:
                            if params not in config[func_name]:
                                config[func_name].append(params)
                        elif params is None:
                            config[func_name] = None
                
                return config
            
            predefined_features_curr_M = create_tsfresh_config_from_feature_list(features_curr_M_list)
            predefined_features_6_M = create_tsfresh_config_from_feature_list(features_6_M_list) 
            predefined_features_12_M = create_tsfresh_config_from_feature_list(features_12_M_list)
            
            rolling_configs = [
                (nb_months * simulation_period_days, "_curr_M", predefined_features_curr_M),
                (6 * nb_months * simulation_period_days, "_6_M", predefined_features_6_M),
                (12 * simulation_period_days, "_12_M", predefined_features_12_M),
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
            demand_y_select = demand_y_select.iloc[:-(nb_months * simulation_period_days_prev)]
            demand_y_select = demand_y_select.set_index('PERFORM_DATE')

            correlation_data = pd.merge(demand_y_select, feat, left_index=True, right_index=True, sort= False)
            correlation_matrix = correlation_data.corr()
            correlation_matrix = correlation_matrix.iloc[:,:1].squeeze()
            selected_columns = correlation_matrix.abs().sort_values(ascending=False).index
            selected_columns = selected_columns.drop(demand_col, errors='ignore')
            selected_columns = selected_columns[:75]
            month_features = pd.DataFrame({
                'feature': selected_columns,
                'year': year,
                'month': month, 
                'feature_rank': range(1, len(selected_columns) + 1)
            })
            all_best_feats.append(month_features)
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
            
            rolling_windows = [30, 60, 90]            
            additional_feat = add_rolling_features(
            demand_X, 
            demand_col=demand_col, 
            windows=rolling_windows)
            additional_feat.set_index('PERFORM_DATE', inplace=True)
            col_name='ACTUAL_DEMAND_rollmean_90'
            nan_dates = additional_feat.index[additional_feat[col_name].isna()]
            if len(nan_dates) > 0:
                min_nan_date = nan_dates.min()
                max_nan_date = nan_dates.max()
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
            
            additional_feat = additional_feat.interpolate(method='spline', order=2, axis=0)
            additional_feat = additional_feat.sort_index()
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
            feat = pd.concat([feat, additional_feat], axis=1)
            feat = feat.dropna(axis=1)

            correlation_data = pd.merge(demand_y[demand_col], feat, left_index=True, right_index=True, sort= False)
            correlation_matrix = correlation_data.corr()
            correlation_matrix = correlation_matrix.iloc[:,:1].squeeze()
            selected_columns = correlation_matrix.abs().sort_values(ascending=False).index
            selected_columns = selected_columns.drop(demand_col, errors='ignore')
            selected_columns = selected_columns[:85]
            feat = feat[selected_columns]

                
            imp_scaler = StandardScaler()
            feat_sc = imp_scaler.fit_transform(feat)
            
            feat_sc_next_days = []
            time_index = pd.DatetimeIndex(feat.index)
            for column in feat_sc.T:
                feat_sc_next_days.append(TimeSeries.from_times_and_values(time_index, column))   
            
            series_names = np.arange(feat_sc.shape[1])
            preds = list()
            
            # Progressive fallback mechanism for dim_x
            successful_forecast = False
            for dim_x in range(initial_dim_x, 0, -1):  # Try from initial_dim_x down to 1
                try:
                    temp_preds = list()
                    for i, series in enumerate(feat_sc_next_days):
                        model = KalmanForecaster(dim_x=dim_x)
                        model.fit(series=series)
                        p = model.predict(simulation_period_days).pd_dataframe().reset_index()
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
            
                if pred_std != ref_std and pred_std > 0: # New ! avant <
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

            col_name = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0_curr_M'
            ref_col = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4_curr_M'
            if col_name in feat_next_days.columns and ref_col in feat_next_days.columns:   
                col_idx = feat_next_days.columns.get_loc(col_name)
                ref_idx = feat_next_days.columns.get_loc(ref_col)
                mask = feat_next_days.iloc[-simulation_period_days:, ref_idx] == 0.0
                feat_next_days.iloc[-simulation_period_days:, col_idx] = np.where(
                    mask,
                    0.0,
                    feat_next_days.iloc[-simulation_period_days:, col_idx]
                )
           
            col_name = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0_12_M'
            ref_col = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4_12_M'
            if col_name in feat_next_days.columns and ref_col in feat_next_days.columns:   
                col_idx = feat_next_days.columns.get_loc(col_name)
                ref_idx = feat_next_days.columns.get_loc(ref_col)
                mask = feat_next_days.iloc[-simulation_period_days:, ref_idx] == 0.0
                feat_next_days.iloc[-simulation_period_days:, col_idx] = np.where(
                    mask,
                    0.0,
                    feat_next_days.iloc[-simulation_period_days:, col_idx]
                )

            col_name = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0_6_M'
            ref_col = demand_col + '__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4_6_M'
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
    
            for col in feat_next_days.columns:
                pred_idx = np.arange(len(feat_next_days))[-simulation_period_days:]
                ref_idx = np.arange(len(feat_next_days))[:-simulation_period_days]
            
                pred_vals = feat_next_days.loc[pred_idx, col]
                ref_max = feat_next_days.loc[ref_idx, col].max()
            
                if ref_max < 0:
                    mask = feat_next_days.loc[pred_idx, col] > 0
                                     
            demand_X = pd.merge(demand_item_fct[['PERFORM_DATE']], feat_next_days, left_index=True, right_index=True, sort=False)
            demand_X = pd.merge(demand_X, X_sup, left_index=True, right_index=True, sort=False)            
            demand_X.set_index('PERFORM_DATE', inplace=True)        
            demand_X = reduce_mem_usage(demand_X)
            
            y_train = forecast_data[['PERFORM_DATE', demand_col]].copy()
            y_train.set_index('PERFORM_DATE', inplace=True)
            
            # Get structural break information from variables defined earlier
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
                X_val = X_train[X_train.shape[0]-2*max(simulation_period_days,simulation_period_days_prev):X_train.shape[0]]
            if nb_months > 1:
                X_train = X_train.iloc[:-2*nb_months*simulation_period_days,:]
            else:
                X_train = X_train.iloc[:-2*max(simulation_period_days,simulation_period_days_prev),:]

            X_bt_train, X_bt_val, y_bt_train, y_bt_val = create_backtest_splits_monthly(
                demand_X, demand_y[demand_col], idx, simulation_period_days, nb_months, simulation_period_days_prev
            )
            if X_bt_train is None or len(X_bt_train) == 0:
                print("Skipping backtesting - not enough historical data")
            else:
                X_bt_train = clean_feature_names(X_bt_train)
                X_bt_val = clean_feature_names(X_bt_val)
                y_bt_train = y_bt_train.values.ravel()
                y_bt_val = y_bt_val.values.ravel()
    
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
            positive_count = monotone_constraints.count(1)
            negative_count = monotone_constraints.count(-1)
            none_count = monotone_constraints.count(0)
            
            import_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\UK\Datas_out\Forecasts\all_params_lot1.csv'
            ff1 = pd.read_csv(import_path, engine='python', doublequote=None, sep=";", encoding='UTF-8')
            import_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\UK\Datas_out\Forecasts\all_params_lot2.csv'
            ff2 = pd.read_csv(import_path, engine='python', doublequote=None, sep=";", encoding='UTF-8')
            params = pd.concat([ff1, ff2], axis=0)
            params['cpt'] = params.groupby(['ITEM_NUMBER_SITE']).cumcount() + 1
            params.drop(columns={'ITEM_NUMBER_SITE', 'Month'}, inplace=True)
            
            # Calculate median AND max parameters for each month (cpt) for hybrid optimization strategy
            param_columns = ['num_leaves', 'max_depth', 'learning_rate', 'n_estimators', 'min_child_samples', 
                           'min_split_gain', 'lambda_l1', 'lambda_l2', 'subsample', 'colsample_bytree', 
                           'feature_fraction', 'bagging_fraction', 'bagging_freq', 'min_data_in_leaf', 'max_bin']
            
            median_params_by_month = {}
            max_params_by_month = {}
            
            for month_idx in range(1, 14):
                month_data = params[params['cpt'] == month_idx]
                if len(month_data) > 0:
                    # Calculate median parameters for months 1-6
                    median_params = {}
                    max_params = {}
                    for param in param_columns:
                        if param in month_data.columns:
                            median_params[param] = month_data[param].median()
                            max_params[param] = month_data[param].max()
                    
                    median_params_by_month[month_idx] = median_params
                    max_params_by_month[month_idx] = max_params
                else:
                    print(f"Warning: No data for month {month_idx}")
            
            # Flag to switch between Optuna optimization and hybrid parameters
            USE_HYBRID_PARAMS = False  # Set to False to use Optuna optimization
            
            def get_optimized_params(month_idx, trial=None):
                if USE_HYBRID_PARAMS:
                    if month_idx <= 6 and month_idx in median_params_by_month:
                        # Use median parameters for early months (stable, accurate)
                        params_dict = median_params_by_month[month_idx].copy()
                        strategy = "median"
                    elif month_idx >= 7 and month_idx in max_params_by_month:
                        # Use max parameters for late months (aggressive, addresses under-forecasting)
                        params_dict = max_params_by_month[month_idx].copy()
                        strategy = "max"
                    else:
                        raise ValueError(f"No hybrid parameters available for month {month_idx}")
                    
                    # Ensure integer parameters are properly typed
                    for int_param in ['num_leaves', 'max_depth', 'n_estimators', 'min_child_samples', 
                                    'bagging_freq', 'min_data_in_leaf', 'max_bin']:
                        if int_param in params_dict:
                            params_dict[int_param] = int(round(params_dict[int_param]))
                    
                    return params_dict
                    
                elif trial is not None:
                    return {
                        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                        'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.0),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 10.0),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 10.0),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                        'max_bin': trial.suggest_int('max_bin', 20, 100)
                    }
                else:
                    raise ValueError(f"No hybrid parameters available for month {month_idx} and no trial provided")
                    
            forecast = []
            
            if USE_HYBRID_PARAMS:
                try:
                    month_for_params = idx + 1
                    best_params = get_optimized_params(month_for_params)
                except (NameError, KeyError) as e:
                    print(f"Month index not available or hybrid params missing, falling back to Optuna: {e}")
                    USE_HYBRID_PARAMS = False
            
            if not USE_HYBRID_PARAMS:
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
                study.optimize(lambda trial: objective_less_than_weekly_LGBM_with_weight(
                trial,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                wgt=wgt,
                enhanced_regularization=production_params['regularization']),
                n_trials=40, callbacks=[print_callback])
                best_params = study.best_params

                del study
                del sampler
                gc.collect()          
            
            X_full = pd.concat([X_train, X_val], axis=0)
            y_full = np.concatenate([y_train, y_val], axis=0)
            model = build_fit_less_than_weekly_LGBMRegressor(
                X_full=X_full,
                y_full=y_full,
                monotone_constraints=monotone_constraints,
                monotone_constraints_method="intermediate",
                **best_params
            )
            pred = model.predict(X_test)
            best_params_df = pd.DataFrame([best_params])
            best_params_df['ITEM_NUMBER_SITE'] = item_id
            best_params_df['Month'] = month
            monthly_best_params.append(best_params_df)
            
            simple_backtest_model = build_fit_less_than_weekly_LGBMRegressor(
                X_full=X_bt_train, 
                y_full=y_bt_train,
                monotone_constraints=monotone_constraints,
                monotone_constraints_method="intermediate",
                **best_params
            )
                
            pred_backtest = simple_backtest_model.predict(X_bt_val)
            y_bt_val_sum = y_bt_val.sum()
            pred_backtest_sum = pred_backtest.sum()
            gc.collect()          
            
            pct1 = len(X_train)/len(X_full)
            pct2 = simulation_period_days/len(X_full)
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
            ci_05 = pd.DataFrame(ci_05, columns=['Forecasted Demand Inf'])
            pred = pd.DataFrame(pred, columns=[demand_col])
            ci_95 = pd.DataFrame(ci_95, columns=['Forecasted Demand Sup'])
                    
            forecast = pd.concat([ci_05, pred, ci_95], axis=1)
            forecast['ITEM_NUMBER_SITE'] = item_id
            forecast = pd.merge(next_days, forecast, left_index=True, right_index=True, sort=False)
                                       
            top_indices = forecast[demand_col].nlargest(demand_days).index
            forecast[demand_col] = np.where(forecast.index.isin(top_indices), forecast[demand_col], 0)
            forecast[demand_col] = forecast[demand_col].abs().astype(int)
            forecast['Forecasted Demand Inf'] = np.where(forecast.index.isin(top_indices), forecast['Forecasted Demand Inf'], 0)
            forecast['Forecasted Demand Inf'] = forecast['Forecasted Demand Inf'].abs().astype(int)
            forecast['Forecasted Demand Sup'] = np.where(forecast.index.isin(top_indices), forecast['Forecasted Demand Sup'], 0)
            forecast['Forecasted Demand Sup'] = forecast['Forecasted Demand Sup'].abs().astype(int)
            forecast['EffectiveDemand'] = np.where(forecast[demand_col]>0,1,0)
            forecast['Year'] = forecast['PERFORM_DATE'].dt.year
            forecast['Month'] = forecast['PERFORM_DATE'].dt.month

            monthly_tot_demand_fct = forecast.groupby(['Year','Month'])[demand_col].sum().reset_index()                                             
                            
            month_sums = forecast[(forecast['Year'] == year) & (forecast['Month'] == month)][demand_col].sum()
            if demand_es_lr_ratio < 40:
                es_lr_threshold = 0.3                                    
            else:
                es_lr_threshold = 0.15                                    
                
            if y_bt_val_sum > 0:
                model_bias_ratio = (pred_backtest_sum - y_bt_val_sum) / y_bt_val_sum  # Positive = overestimate, Negative = underestimate
                abs_error_ratio = abs(model_bias_ratio)
                
                # === YoY TREND ANALYSIS ===
                # Check for year-over-year trend patterns independently of bias correction
                prev_year = year - 1
                same_month_prev_year = monthly_tot_demand[(monthly_tot_demand['Year'] == prev_year) & (monthly_tot_demand['Month'] == month)]                
                if not same_month_prev_year.empty:
                    prev_year_demand = same_month_prev_year[demand_col].iloc[0]
                    
                    if prev_year_demand > 0:
                        # Calculate YoY growth rates
                        lgbm_yoy_growth = (month_sums - prev_year_demand) / prev_year_demand
                        monthly_forecast_yoy_growth = (demand_monthly_forecast_w - prev_year_demand) / prev_year_demand
                        
                        # Check for significant YoY trend discrepancy
                        yoy_growth_diff = abs(monthly_forecast_yoy_growth - lgbm_yoy_growth)
                        
                        # === PROGRESSIVE THRESHOLD STRATEGY ===
                        # Make thresholds more aggressive as forecast horizon increases
                        month_horizon = idx + 1  # idx is 0-based, so month 1, 2, 3...
                        
                        # === ITEM-SPECIFIC VOLATILITY ADJUSTMENT ===
                        # Calculate historical volatility for this item
                        if len(monthly_tot_demand) > 6:  # Need sufficient history
                            monthly_demands = monthly_tot_demand[demand_col]
                            monthly_cv = monthly_demands.std() / monthly_demands.mean() if monthly_demands.mean() > 0 else 1.0
                            
                            # Volatility-based threshold adjustment
                            if monthly_cv < 0.3:      # Low volatility item (predictable)
                                volatility_factor = 0.8   # Stricter thresholds (trust LightGBM more)
                                volatility_type = "Low volatility"
                            elif monthly_cv > 0.8:    # High volatility item (erratic)
                                volatility_factor = 1.2   # Looser thresholds (trust ES/LR more)
                                volatility_type = "High volatility"
                            else:                      # Medium volatility
                                volatility_factor = 1.0   # Standard thresholds
                                volatility_type = "Medium volatility"
                                
                        else:
                            volatility_factor = 1.0
                        
                        # === LIFE CYCLE STAGE DETECTION ===
                        # Detect if item is in growth, mature, or declining phase
                        if len(monthly_tot_demand) > 12:  # Need at least 1 year of data
                            monthly_demands = monthly_tot_demand[demand_col].values
                            
                            # Calculate trend over recent periods
                            recent_6_months = monthly_demands[-6:].mean() if len(monthly_demands) >= 6 else monthly_demands[-3:].mean()
                            older_6_months = monthly_demands[-12:-6].mean() if len(monthly_demands) >= 12 else monthly_demands[:-3].mean()
                            
                            if older_6_months > 0:
                                trend_ratio = recent_6_months / older_6_months
                                
                                if trend_ratio > 1.2:         # Growing significantly
                                    lifecycle_stage = "growth"
                                    lifecycle_factor = 0.9      # Trust LightGBM slightly more for growth
                                elif trend_ratio < 0.8:       # Declining significantly  
                                    lifecycle_stage = "declining"
                                    lifecycle_factor = 1.1      # Trust ES/LR more for decline
                                else:                          # Stable/mature
                                    lifecycle_stage = "mature"
                                    lifecycle_factor = 0.95     # Slight preference for LightGBM
                                    
                            else:
                                lifecycle_factor = 1.0
                        else:
                            lifecycle_factor = 1.0
                        
                        # Base thresholds by forecast horizon
                        if month_horizon <= 3:
                            base_yoy_threshold = 0.4      # Conservative for near-term (months 1-3)
                            base_blend_weight_cap = 0.5
                            horizon_type = "conservative"
                        elif month_horizon <= 6:
                            base_yoy_threshold = 0.3      # More aggressive for mid-term (months 4-6)
                            base_blend_weight_cap = 0.6
                            horizon_type = "moderate"
                        else:
                            base_yoy_threshold = 0.15      # Very aggressive for long-term (months 7-13)
                            base_blend_weight_cap = 0.8
                            horizon_type = "aggressive"
                        
                        # Apply volatility and lifecycle adjustments
                        combined_factor = volatility_factor * lifecycle_factor
                        yoy_threshold = base_yoy_threshold * combined_factor
                        
                        # Blend weight adjustment: lower factor = trust LightGBM more, higher factor = trust ES/LR more
                        blend_weight_adjustment = (combined_factor - 1.0) * 0.1  # Scale the adjustment
                        blend_weight_cap = min(0.8, max(0.3, base_blend_weight_cap + blend_weight_adjustment))  # Cap between 0.3-0.8
                                                
                        # Trigger YoY correction using adaptive thresholds
                        if (yoy_growth_diff > yoy_threshold) or (monthly_forecast_yoy_growth > 1.0 and lgbm_yoy_growth < 0.2):
                            
                            # Use blended approach: weight toward monthly forecast if it shows strong growth trend
                            if monthly_forecast_yoy_growth > lgbm_yoy_growth + 0.3:  # Monthly shows much stronger growth
                                blend_weight = min(blend_weight_cap, yoy_growth_diff)  # Use progressive weight cap
                                target_monthly_demand = int(month_sums * (1 - blend_weight) + demand_monthly_forecast_w * blend_weight)
                                
                                # Apply YoY trend correction
                                diff = target_monthly_demand - month_sums
                                if abs(diff) > max(5, month_sums * 0.03):  # More sensitive threshold for trend correction
                                    if diff > 0:  # Need to increase
                                        pos_indices = forecast[(forecast['Month'] == month) & (forecast[demand_col] > 0)].index
                                        n_pos = len(pos_indices)
                                        if n_pos > 0:
                                            add_per_day = int(np.ceil(diff / n_pos))
                                            forecast.loc[pos_indices, demand_col] += add_per_day
                                            forecast.loc[pos_indices, 'Forecasted Demand Inf'] += add_per_day
                                            forecast.loc[pos_indices, 'Forecasted Demand Sup'] += add_per_day
                                        elif target_monthly_demand > 0:  # No positive days but need to add demand
                                            possible_days = forecast[(forecast['Month'] == month)].index
                                            if len(possible_days) >= 21:
                                                selection_count = min(max(1, demand_days), len(possible_days) - 5)
                                                random.seed(RANDOM_SEED)

                                                selected_indices = random.sample(list(possible_days)[5:21], selection_count)
                                                add_per_day = int(np.ceil(target_monthly_demand / len(selected_indices)))
                                                for idx in selected_indices:
                                                    forecast.at[idx, demand_col] = add_per_day
                                                    forecast.at[idx, 'Forecasted Demand Inf'] = int(add_per_day * 0.95)
                                                    forecast.at[idx, 'Forecasted Demand Sup'] = int(add_per_day * 1.15)
                                    # Update month_sums for subsequent bias correction
                                    month_sums = forecast[(forecast['Year'] == year) & (forecast['Month'] == month)][demand_col].sum()
                        else:
                            print("YoY trends are aligned - no trend correction needed")
                    else:
                        print(f"Previous year {prev_year}-{month:02d}: No demand for comparison")
                else:
                    print(f"No data available for {prev_year}-{month:02d} YoY comparison")
                
                # Only apply correction if it has significant bias (>10%) and monthly forecasts disagree
                if abs_error_ratio > 0.1 and month_sums > 0:
                    # Determine target based on model bias and monthly forecast
                    if model_bias_ratio > es_lr_threshold:  # Model overestimates
                        # Use the lower of monthly forecast or bias-corrected LightGBM
                        target_monthly_demand = min(demand_monthly_forecast_w, month_sums * (1 - model_bias_ratio))
                    else:  # Model underestimates  
                        # Take the higher of monthly forecast or bias-corrected LightGBM
                        target_monthly_demand = max(demand_monthly_forecast_w, month_sums * (1 - model_bias_ratio))
                    if target_monthly_demand > 0 and demand_days == 0:
                        if target_monthly_demand <= 20:
                            demand_days = 1
                        elif target_monthly_demand <= 100:
                            demand_days = 2 
                        else:
                            demand_days = 5
                    
                    # Apply correction if there is still a significant difference
                    diff = target_monthly_demand - month_sums
                    if abs(diff) > max(10, month_sums * 0.05):
                        
                        if diff > 0:  # It need to increase
                            pos_indices = forecast[(forecast['Month'] == month) & (forecast[demand_col] > 0)].index
                            n_pos = len(pos_indices)
                            distribute_monthly_demand(forecast, target_monthly_demand, demand_days, month, demand_col, diff)                        
                        else:  # It need to decrease (diff < 0)
                            reduction_factor = target_monthly_demand / month_sums if month_sums > 0 else 0
                            pos_indices = forecast[(forecast['Month'] == month) & (forecast[demand_col] > 0)].index
                            if len(pos_indices) > 0:
                                forecast.loc[pos_indices, demand_col] = (forecast.loc[pos_indices, demand_col] * reduction_factor).round().astype(int)
                                forecast.loc[pos_indices, 'Forecasted Demand Inf'] = (forecast.loc[pos_indices, 'Forecasted Demand Inf'] * reduction_factor).round().astype(int)
                                forecast.loc[pos_indices, 'Forecasted Demand Sup'] = (forecast.loc[pos_indices, 'Forecasted Demand Sup'] * reduction_factor).round().astype(int)
                                print(f"Reduced demand by factor {reduction_factor:.3f}")
                    else:
                        print("No correction needed - difference within acceptable range")
                elif month_sums == 0 and abs_error_ratio > 0.02:
                    if demand_monthly_forecast_w > 0:
                        target_monthly_demand = int(demand_monthly_forecast_w)
                        possible_days = forecast[(forecast['Month'] == month)].index
                        if len(possible_days) >= 21:
                            if demand_days == 0:
                                # Use reasonable default when days=0 but volume>0
                                selection_count = min(3, len(possible_days) - 5)
                            else:
                                selection_count = min(demand_days, len(possible_days) - 5)
                            
                            if selection_count > 0:
                                random.seed(RANDOM_SEED)

                                selected_indices = random.sample(list(possible_days)[5:21], selection_count)
                                add_per_day = int(np.ceil(target_monthly_demand / len(selected_indices)))
                                for idx in selected_indices:
                                    forecast.at[idx, demand_col] = add_per_day
                                    forecast.at[idx, 'Forecasted Demand Inf'] = int(add_per_day * 0.95)
                                    forecast.at[idx, 'Forecasted Demand Sup'] = int(add_per_day * 1.15)
                            else:
                                print("Cannot distribute demand - insufficient days available")                        
            else:
                if demand_monthly_forecast_w > 0 and demand_days > 0:
                    target_monthly_demand = int(demand_monthly_forecast_w)
                    possible_days = forecast[(forecast['Month'] == month)].index
                    if len(possible_days) >= 21:
                        random.seed(RANDOM_SEED)

                        selected_indices = random.sample(list(possible_days)[5:21], min(demand_days, len(possible_days)-5))
                        add_per_day = int(np.ceil(target_monthly_demand / len(selected_indices)))
                        for idx in selected_indices:
                            forecast.at[idx, demand_col] = add_per_day
                            forecast.at[idx, 'Forecasted Demand Inf'] = int(add_per_day * 0.95)
                            forecast.at[idx, 'Forecasted Demand Sup'] = int(add_per_day * 1.15)
            
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
            monthly_tot_demand = pd.concat([monthly_tot_demand, monthly_tot_demand_fct], axis=0)
            
        # Check if forecast_data was created in any of the conditional blocks
        if forecast_data is None:
            print(f"ERROR: No forecast generated for item {item_id} with category {actual_category}")
            print("Available categories: Daily, Weekly, Monthly, Bi-annual")
            continue
                
        forecast_data['Forecasted Demand Inf'] = np.minimum(
            forecast_data['Forecasted Demand Inf'],
            forecast_data['Forecasted Demand Sup']
        )
        forecast_data['Forecasted Demand Sup'] = np.maximum.reduce([
            forecast_data['Forecasted Demand Sup'],
            forecast_data['Forecasted Demand Inf'],
            forecast_data[demand_col]
        ])
        forecast_data['EffectiveDemand'] = np.where(forecast_data['ACTUAL_DEMAND'] > 0,1,0)
        forecast_data['Forecasted Demand Inf'] = forecast_data['Forecasted Demand Inf'].astype(float)
        forecast_data['Forecasted Demand Sup'] = forecast_data['Forecasted Demand Sup'].astype(float)
        mask = (forecast_data['ACTUAL_DEMAND'] > 0) & (forecast_data['ACTUAL_DEMAND'] < forecast_data['Forecasted Demand Inf'])
        forecast_data.loc[mask, 'Forecasted Demand Inf'] = forecast_data.loc[mask, 'ACTUAL_DEMAND'] * 0.95
        forecast_data.loc[mask, 'Forecasted Demand Sup'] = forecast_data.loc[mask, 'ACTUAL_DEMAND'] * 1.15
        forecast_data['Forecasted Demand Inf'] = forecast_data['Forecasted Demand Inf'].round(0)
        forecast_data['Forecasted Demand Sup'] = forecast_data['Forecasted Demand Sup'].round(0)
        if np.issubdtype(forecast_data['Forecasted Demand Inf'].dtype, np.floating):
            forecast_data['Forecasted Demand Inf'] = forecast_data['Forecasted Demand Inf'].astype('Int64')
        if np.issubdtype(forecast_data['Forecasted Demand Sup'].dtype, np.floating):
            forecast_data['Forecasted Demand Sup'] = forecast_data['Forecasted Demand Sup'].astype('Int64')
        # === STRUCTURAL BREAK CONFIDENCE INTERVAL ADJUSTMENT ===
        # Apply confidence multiplier if structural breaks were detected
        confidence_multiplier = getattr(forecast_data, 'attrs', {}).get('confidence_multiplier', 1.0)
        break_info = getattr(forecast_data, 'attrs', {}).get('break_info', None)
        
        if confidence_multiplier > 1.0:
            
            # Expand confidence intervals around the central forecast
            central_forecast = forecast_data[demand_col]
            lower_diff = central_forecast - forecast_data['Forecasted Demand Inf']
            upper_diff = forecast_data['Forecasted Demand Sup'] - central_forecast
            
            # Apply multiplier to the differences (expanding uncertainty)
            forecast_data['Forecasted Demand Inf'] = central_forecast - (lower_diff * confidence_multiplier)
            forecast_data['Forecasted Demand Sup'] = central_forecast + (upper_diff * confidence_multiplier)
            
            # Ensure non-negative intervals
            forecast_data['Forecasted Demand Inf'] = np.maximum(forecast_data['Forecasted Demand Inf'], 0)        
    
        true_forecast = forecast_data[forecast_data['PERFORM_DATE'] > last_date_hist].copy()
        global_interval = simulate_aggregate_intervals(true_forecast, dist='nbinom')
        global_interval['ITEM_NUMBER_SITE'] = item_id
        # Handle edge case where all forecast values are zero (common for bi-annual items)
        if global_interval[demand_col].iloc[0] == 0:
            global_interval['Ratio1'] = 1.0  # Default conservative ratio
            global_interval['Ratio2'] = 1.0  # Default conservative ratio
        else:
            global_interval['Ratio1'] = global_interval['Forecasted Demand Inf'] / global_interval[demand_col]
            global_interval['Ratio2'] = global_interval['Forecasted Demand Sup'] / global_interval[demand_col]
        last_date = forecast_data['PERFORM_DATE'].max()
        start_date = last_date - pd.DateOffset(months=horizon)
        yearly_tot = forecast_data.loc[
            (forecast_data['PERFORM_DATE'] > start_date) & (forecast_data['PERFORM_DATE'] <= last_date),
            demand_col
        ].sum()
        last_year = forecast_data.loc[
            (forecast_data['PERFORM_DATE'] > start_date) & (forecast_data['PERFORM_DATE'] <= last_date)
        ]
    
        global_interval['Forecasted Demand Inf'] = global_interval['Ratio1'] * yearly_tot
        global_interval['Forecasted Demand Sup'] = global_interval['Ratio2'] * yearly_tot
        
        # Handle NaN and infinite values before converting to int
        global_interval['Forecasted Demand Inf'] = global_interval['Forecasted Demand Inf'].abs()
        global_interval['Forecasted Demand Sup'] = global_interval['Forecasted Demand Sup'].abs()
        
        # Replace NaN and infinite values with 0 before converting to int
        global_interval['Forecasted Demand Inf'] = global_interval['Forecasted Demand Inf'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        global_interval['Forecasted Demand Sup'] = global_interval['Forecasted Demand Sup'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
        global_interval['Forecasted Demand'] = yearly_tot
        global_interval = global_interval[['ITEM_NUMBER_SITE', 'Forecasted Demand', 'Forecasted Demand Inf', 'Forecasted Demand Sup']]
        
        forecast_data['Week'] = ((forecast_data['PERFORM_DATE'] - forecast_data['PERFORM_DATE'].dt.to_period('M').dt.start_time).dt.days // 7) + 1
        forecast_data['Week'] = forecast_data['Week'].astype('Int64')
    
        forecast_results_summary = forecast_data.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month']).agg(
            Nb_Days_sum=('EffectiveDemand', 'sum'),
            Forecasted_Demand_sum=(demand_col, 'sum'),
            Forecasted_Demand_mean=(demand_col, 'mean'),
            Forecasted_Demand_std=(demand_col, 'std'),
            Forecasted_Demand_inf_sum=('Forecasted Demand Inf', 'sum'),
            Forecasted_Demand_inf_mean=('Forecasted Demand Inf', 'mean'),
            Forecasted_Demand_inf_std=('Forecasted Demand Inf', 'std'),
            Forecasted_Demand_sup_sum=('Forecasted Demand Sup', 'sum'),
            Forecasted_Demand_sup_mean=('Forecasted Demand Sup', 'mean'),
            Forecasted_Demand_sup_std=('Forecasted Demand Sup', 'std'),
        ).reset_index()
    
        forecast_results_summary.rename(columns={
            'Nb_Days_sum': 'Number of Days of Demand',
            'Forecasted_Demand_sum': 'Forecasted Demand',
            'Forecasted_Demand_mean': 'Forecasted Demand mean',
            'Forecasted_Demand_std': 'Forecasted Demand std',
            'Forecasted_Demand_inf_sum': 'Forecasted Demand Inf',
            'Forecasted_Demand_inf_mean': 'Forecasted Demand Inf mean',
            'Forecasted_Demand_inf_std': 'Forecasted Demand Inf std',
            'Forecasted_Demand_sup_sum': 'Forecasted Demand Sup',
            'Forecasted_Demand_sup_mean': 'Forecasted Demand Sup mean',
            'Forecasted_Demand_sup_std': 'Forecasted Demand Sup std',
        }, inplace=True)
    
        forecast_results_summary['Forecasted Demand mean'] = forecast_results_summary['Forecasted Demand mean'].round(3)
        forecast_results_summary['Forecasted Demand std'] = forecast_results_summary['Forecasted Demand std'].round(3)
        forecast_results_summary['Forecasted Demand Inf mean'] = forecast_results_summary['Forecasted Demand Inf mean'].round(3)
        forecast_results_summary['Forecasted Demand Inf std'] = forecast_results_summary['Forecasted Demand Inf std'].round(3)
        forecast_results_summary['Forecasted Demand Sup mean'] = forecast_results_summary['Forecasted Demand Sup mean'].round(3)
        forecast_results_summary['Forecasted Demand Sup std'] = forecast_results_summary['Forecasted Demand Sup std'].round(3)

    if item_selection.loc[item_selection['ITEM_NUMBER_SITE'] == item_id, 
                          'PURCHASE_CYCLE_CATEGORY'].iloc[0] in ["Monthly"]:    
        all_global_interval.append(global_interval)
        all_forecasts.append(forecast_data)
        all_summaries.append(forecast_results_summary)
        all_last_years.append(last_year)
        all_best_params.extend(monthly_best_params)
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
        
        try:
            del feat, demand_X, demand_y, feat_next_days
            del X_train, y_train, X_test, X_val
            del X_bt_train, X_bt_val, y_bt_train, y_bt_val
            del forecast_data, demand_item_fct
        except:
            pass
        gc.collect()

#final_feats = pd.concat(all_best_feats, ignore_index=True)
final_forecast = pd.concat(all_forecasts, ignore_index=True)
weekly_grouped = final_forecast.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month', 'Week']).agg({'ACTUAL_DEMAND': 'sum',
                                                                                    'EffectiveDemand': 'sum'}).reset_index()
monthly_grouped = final_forecast.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month']).agg({'ACTUAL_DEMAND': 'sum',
                                                                                    'EffectiveDemand': 'sum'}).reset_index()

final_summary = pd.concat(all_summaries, ignore_index=True)
final_intervals = pd.concat(all_global_interval, ignore_index=True)
final_last_years = pd.concat(all_last_years, ignore_index=True)
final_forecast.drop(columns={'DaysInMonth', 'EffectiveNbDays', 'EffectiveDemand', 'Week2', 'History'}, inplace=True)


duration = (time.time() - start_time)/60

#del final_monthly_summary

export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_summaries.csv'
final_summary.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_forecast_data.csv'
final_forecast.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_intervals.csv'
final_intervals.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_monthly_summaries.csv'
monthly_grouped.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')


###############  Bi-annual
final_forecast_bi_annual = pd.DataFrame()
final_intervals_bi_annual = pd.DataFrame()
all_forecasts = []
all_summaries = []
all_global_interval = []
all_last_years = []

for item_id in item_selection_bi_annual['ITEM_NUMBER_SITE']:
    # Debug: Check what category this item actually has
    actual_category = item_selection_bi_annual.loc[item_selection_bi_annual['ITEM_NUMBER_SITE'] == item_id, 'PURCHASE_CYCLE_CATEGORY'].iloc[0]
    print(f"Processing item {item_id} with category: {actual_category}")
    
    forecast_data = None
    
    demand_item_fct = demand_period_long[demand_period_long['ITEM_NUMBER_SITE'] == item_id].copy()  
    demand_item_fct.reset_index(drop=True, inplace=True)
    demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
    demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month
    demand_item_fct['DaysInMonth'] = demand_item_fct['PERFORM_DATE'].dt.days_in_month
    demand_item_fct['EffectiveNbDays'] = 1
    demand_item_fct['EffectiveDemand'] = np.where(demand_item_fct[demand_col] > 0, 1, 0)
    monthly_tot = demand_item_fct.groupby(['Year','Month']).agg({'DaysInMonth':'max',                                                                 
                                                              'EffectiveNbDays': 'sum',
                                                              'EffectiveDemand': 'sum',
                                                              demand_col: 'sum'}).reset_index()
    if (
    (monthly_tot.iloc[-1]['EffectiveNbDays'] < 28 and monthly_tot.iloc[-1]['Month'] == 2) or
    (monthly_tot.iloc[-1]['EffectiveNbDays'] < 30 and monthly_tot.iloc[-1]['Month'] != 2)
    ):
        monthly_tot = monthly_tot.iloc[:-1]
        
    demand_item_fct = pd.merge(demand_item_fct, monthly_tot[['Year','Month']], on=['Year','Month'], how='inner', sort=False)


    if item_selection_bi_annual.loc[item_selection_bi_annual['ITEM_NUMBER_SITE'] == item_id, 'PURCHASE_CYCLE_CATEGORY'].iloc[0] in ["Bi-annual"]:
    
        #    demand_col = 'Forecast_Seasonal_Regime'
        start_time = time.time()

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
        demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
        demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month        
        existing_days = demand_item_fct[
            (demand_item_fct['Year'] == last_year) & (demand_item_fct['Month'] == last_month)
        ]['PERFORM_DATE']
        missing_days = set(all_days) - set(existing_days)

        if missing_days:
            missing_rows = [{'ITEM_NUMBER_SITE': item_id, 'PERFORM_DATE': date} for date in sorted(missing_days)]
            missing_df = pd.DataFrame(missing_rows)
            demand_item_fct = pd.concat([demand_item_fct, missing_df], ignore_index=True, sort=False)
            demand_item_fct.sort_values('PERFORM_DATE', inplace=True)
            demand_item_fct.reset_index(drop=True, inplace=True)
            demand_item_fct['Year'] = demand_item_fct['PERFORM_DATE'].dt.year
            demand_item_fct['Month'] = demand_item_fct['PERFORM_DATE'].dt.month        

        #cut_off = pd.to_datetime('2025-05-31 00:00:00', format=ts_format_2)
        #demand_item_fct = demand_item_fct[demand_item_fct['PERFORM_DATE'] <= cut_off].copy()

        days = demand_item_fct[['PERFORM_DATE']].copy()
        days['ITEM_NUMBER_SITE'] = item_id
        days['DayOfYear'] = days['PERFORM_DATE'].dt.dayofyear

        demand_item_fct['MonthPeriod'] = demand_item_fct['PERFORM_DATE'].dt.to_period('M')
        demand_item_fct['FirstDayOfMonth'] = demand_item_fct['MonthPeriod'].dt.start_time
        demand_item_fct['Week'] = ((demand_item_fct['PERFORM_DATE'] - demand_item_fct['FirstDayOfMonth']).dt.days // 7) + 1
        demand_item_fct['Week'] = demand_item_fct['Week'].astype('Int64')
        demand_item_fct.drop(['MonthPeriod', 'FirstDayOfMonth'], axis=1, inplace=True)
        demand_item_fct['EffectiveNbDays'] = 1

        first_future_date = demand_item_fct['PERFORM_DATE'].max() - pd.DateOffset(months=6) + pd.Timedelta(days=1)
        first_future_month = first_future_date.to_period('M')        
        mask_future = demand_item_fct['PERFORM_DATE'] > last_date

        demand_item_fct.loc[mask_future, 'Week2'] = (
            (demand_item_fct.loc[mask_future, 'PERFORM_DATE'].dt.day - 1) // 7 + 1
        )

        demand_item_fct['Week2'] = demand_item_fct['Week2'].astype('Int64')

        future_months = (demand_item_fct.loc[mask_future, ['Year', 'Month']].drop_duplicates()
                         .sort_values(['Year', 'Month'])
                         .reset_index(drop=True))
        next_months = {(row['Year'], row['Month']): [1] for _, row in future_months.iterrows()}

        forecast_data = demand_item_fct.copy()
        forecast_data = forecast_data[~pd.isna(forecast_data[demand_col])].copy()
        for col in ['Forecasted Demand Inf', 'Forecasted Demand Sup']:
            if col not in forecast_data.columns:
                forecast_data[col] = np.nan                
        forecast_data['History'] = 1

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

        monthly_tot = demand_item_fct.groupby(['Year','Month']).agg({'EffectiveNbDays': 'sum', #'Day Name':'first',                                                                 
                                                                  'PERFORM_DATE': ['min', 'max']}).reset_index()
        monthly_tot.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in monthly_tot.columns]
        monthly_tot.rename(columns={'PERFORM_DATE_min': 'PERFORM_DATE', 'EffectiveNbDays_sum': 'EffectiveNbDays'}, inplace=True)
        monthly_tot = monthly_tot.sort_values('PERFORM_DATE').reset_index(drop=True)

        monthly_tot_demand = forecast_data.groupby(['Year','Month'])[demand_col].sum().reset_index()                                             

        monthly_forecast=[]

        historical_demand = forecast_data[demand_col].dropna()
        y_T = historical_demand.iloc[-1]  # Last observed value
        y_1 = historical_demand.iloc[0]   # First observed value  
        T = len(historical_demand)        # Number of historical observations
        y_mean = historical_demand.mean() # Historical average

        last_date = forecast_data['PERFORM_DATE'].max()
        forecast_start = last_date + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.DateOffset(months=13) - pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')

        naive_forecasts = pd.DataFrame({
            'PERFORM_DATE': forecast_dates,
            'ITEM_NUMBER_SITE': item_id
        })

        # Method 1: Mean Method - ŷ = (1/T) ∑ y_t
        naive_forecasts['Forecast_Mean'] = y_mean


        hist_monthly = forecast_data.groupby(['Year', 'Month'])[demand_col].sum().reset_index()
        peak_months = hist_monthly[hist_monthly[demand_col] > 0]['Month'].value_counts().index.tolist()
        biannual_months = sorted(peak_months)
        #order_sizes = forecast_data[(forecast_data['Month'].isin(biannual_months)) & (forecast_data[demand_col] > 0)][demand_col]
        order_sizes = forecast_data[(forecast_data['Month'].isin(peak_months)) & (forecast_data[demand_col] > 0)][demand_col]
        if len(order_sizes) > 0:
            typical_order_size = int(round(order_sizes.median()))
        else:
            typical_order_size = 1  # fallback

        # Only forecast in the two peak months, on the 15th of each
        robust_biannual = []
        for date in forecast_dates:
            if date.month in biannual_months and date.day == 15:
                robust_biannual.append(typical_order_size)
            else:
                robust_biannual.append(0)
        naive_forecasts['Forecast_Naive'] = robust_biannual
        #print("[DEBUG] Forecast_Naive (robust bi-annual) assigned. Sample:")
        #print(naive_forecasts[['PERFORM_DATE','Forecast_Naive']].head(31))

        # Method 3: Seasonal Naïve Method - ŷ = y_{same_day_last_year}
        seasonal_naive = []
        for date in forecast_dates:
            try:
                # Try to find exact date one year ago
                same_date_last_year = date - pd.DateOffset(years=1)
                if same_date_last_year in forecast_data['PERFORM_DATE'].values:
                    seasonal_value = forecast_data[forecast_data['PERFORM_DATE'] == same_date_last_year][demand_col].iloc[0]
                else:
                    window_start = same_date_last_year - pd.Timedelta(days=3)
                    window_end = same_date_last_year + pd.Timedelta(days=3)
                    window_data = forecast_data[
                        (forecast_data['PERFORM_DATE'] >= window_start) & 
                        (forecast_data['PERFORM_DATE'] <= window_end)
                    ][demand_col]
                    seasonal_value = window_data.mean() if len(window_data) > 0 else y_mean
                seasonal_naive.append(seasonal_value)
            except:
                seasonal_naive.append(y_mean)  # Fallback to mean

        naive_forecasts['Forecast_Seasonal'] = seasonal_naive

        # Method 4: Drift Method - ŷ = y_T + (h/(T-1))(y_T - y_1)
        drift_slope = (y_T - y_1) / (T - 1) if T > 1 else 0
        drift_forecasts = []
        for h in range(1, len(forecast_dates) + 1):
            drift_value = y_T + h * drift_slope
            drift_forecasts.append(max(0, drift_value))

        naive_forecasts['Forecast_Drift'] = drift_forecasts

        def select_best_naive_method(historical_data, break_info_result=None):
            selection_info = {
                'method': 'ensemble',
                'weights': {'mean': 0.25, 'naive': 0.25, 'seasonal': 0.25, 'drift': 0.25},
                'reasoning': [],
                'regime_analysis': {}
            }
            # Analyze historical patterns
            hist_with_dates = forecast_data[['PERFORM_DATE', demand_col]].dropna()
            hist_with_dates['Year'] = hist_with_dates['PERFORM_DATE'].dt.year
            hist_with_dates['Month'] = hist_with_dates['PERFORM_DATE'].dt.month
            
            def detect_regime_break(data):
                if len(data) < 24:  # Need at least 2 years of data
                    return None, 0.0, data, pd.DataFrame()
                
                years = sorted(data['Year'].unique())
                if len(years) < 3:  # Need at least 3 years for meaningful break detection
                    return None, 0.0, data, pd.DataFrame()
                
                best_break_year = None
                best_confidence = 0.0
                best_pre_regime = data
                best_post_regime = pd.DataFrame()
                
                # Test each year as a potential break point (minus first and last year)
                for test_year in years[1:-1]:
                    pre_data = data[data['Year'] < test_year]
                    post_data = data[data['Year'] >= test_year]
                    
                    # Need sufficient data in both regimes
                    if len(pre_data) < 12 or len(post_data) < 6:
                        continue
                    
                    # Calculate regime characteristics
                    pre_mean = pre_data[demand_col].mean()
                    post_mean = post_data[demand_col].mean()
                    pre_std = pre_data[demand_col].std()
                    post_std = post_data[demand_col].std()
                    
                    # Calculate confidence metrics
                    mean_change = abs(post_mean - pre_mean) / (pre_mean + 1e-6)
                    
                    # Frequency analysis
                    pre_nonzero_freq = len(pre_data[pre_data[demand_col] > 0]) / len(pre_data)
                    post_nonzero_freq = len(post_data[post_data[demand_col] > 0]) / len(post_data)
                    freq_change = abs(post_nonzero_freq - pre_nonzero_freq)
                    
                    # Order size analysis
                    pre_orders = pre_data[pre_data[demand_col] > 0][demand_col]
                    post_orders = post_data[post_data[demand_col] > 0][demand_col]
                    
                    order_change = 0.0
                    if len(pre_orders) > 0 and len(post_orders) > 0:
                        pre_order_mean = pre_orders.mean()
                        post_order_mean = post_orders.mean()
                        order_change = abs(post_order_mean - pre_order_mean) / (pre_order_mean + 1e-6)

                    std_change = 0.0
                    if pre_std > 0:
                        std_change = abs(post_std - pre_std) / (pre_std + 1e-6)
                    
                    # Combine confidence metrics
                    confidence = (mean_change * 0.35 + freq_change * 0.25 + order_change * 0.25 + std_change * 0.15)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_break_year = test_year
                        best_pre_regime = pre_data
                        best_post_regime = post_data
                
                return best_break_year, best_confidence, best_pre_regime, best_post_regime
            
            def detect_seasonal_patterns(data, regime_name=""):
                if len(data) == 0:
                    return [], 0.0, "none"
                
                monthly_demand = data.groupby('Month')[demand_col].sum()
                monthly_frequency = data[data[demand_col] > 0].groupby('Month').size()
                
                # Find months with significant demand
                peak_months = []
                if len(monthly_frequency) > 0:
                    # Use frequency-based detection (months with orders)
                    peak_months = monthly_frequency[monthly_frequency >= 1].index.tolist()
                
                # Calculate pattern strength
                pattern_strength = 0.0
                if len(peak_months) > 0:
                    total_demand = monthly_demand.sum()
                    peak_demand = monthly_demand[peak_months].sum()
                    pattern_strength = peak_demand / (total_demand + 1e-6)
                
                # Determine pattern type
                pattern_type = "none"
                if len(peak_months) == 2:
                    gap = abs(peak_months[1] - peak_months[0])
                    if gap in [2, 3, 4, 5, 6, 7]:  # 2-7 months apart (flexible bi-annual/quarterly)
                        pattern_type = "bi_annual"
                    elif gap == 9:  # 3 months apart
                        pattern_type = "quarterly"
                elif len(peak_months) == 1:
                    pattern_type = "annual"
                elif len(peak_months) >= 3:
                    pattern_type = "multiple"
                
                return peak_months, pattern_strength, pattern_type
            
            def detect_order_sizes(data, regime_name=""):
                if len(data) == 0:
                    return [], 0, 0
                
                orders = data[data[demand_col] > 0][demand_col]
                if len(orders) == 0:
                    return [], 0, 0
                
                # Find most common order sizes
                size_counts = orders.value_counts()
                common_sizes = size_counts.index.tolist()[:3]  # Top 3 sizes
                
                primary_size = common_sizes[0] if len(common_sizes) > 0 else 0
                secondary_size = common_sizes[1] if len(common_sizes) > 1 else 0
                
                return common_sizes, primary_size, secondary_size
            
            # DYNAMIC REGIME ANALYSIS
            regime_break_year, regime_confidence, pre_regime_data, post_regime_data = detect_regime_break(hist_with_dates)
            
            # Store dynamic regime info
            selection_info['regime_analysis']['break_year'] = regime_break_year
            selection_info['regime_analysis']['break_confidence'] = regime_confidence
            selection_info['regime_analysis']['pre_regime_samples'] = len(pre_regime_data)
            selection_info['regime_analysis']['post_regime_samples'] = len(post_regime_data)
            
            # Dynamic seasonal pattern detection
            pre_regime_patterns = detect_seasonal_patterns(pre_regime_data, "pre_regime")
            post_regime_patterns = detect_seasonal_patterns(post_regime_data, "post_regime")
            
            selection_info['regime_analysis']['pre_regime_peaks'] = pre_regime_patterns[0]
            selection_info['regime_analysis']['pre_regime_pattern_strength'] = pre_regime_patterns[1]
            selection_info['regime_analysis']['pre_regime_pattern_type'] = pre_regime_patterns[2]
            
            selection_info['regime_analysis']['post_regime_peaks'] = post_regime_patterns[0]
            selection_info['regime_analysis']['post_regime_pattern_strength'] = post_regime_patterns[1]
            selection_info['regime_analysis']['post_regime_pattern_type'] = post_regime_patterns[2]
            
            # Dynamic order size detection
            pre_regime_orders = detect_order_sizes(pre_regime_data, "pre_regime")
            post_regime_orders = detect_order_sizes(post_regime_data, "post_regime")
            
            selection_info['regime_analysis']['pre_regime_order_sizes'] = pre_regime_orders[0]
            selection_info['regime_analysis']['pre_regime_primary_size'] = pre_regime_orders[1]
            selection_info['regime_analysis']['post_regime_order_sizes'] = post_regime_orders[0]
            selection_info['regime_analysis']['post_regime_primary_size'] = post_regime_orders[1]
            
            if regime_break_year and regime_confidence > 0.15:
                previous_year = pre_regime_data
                regime_post_break = post_regime_data
                selection_info['regime_analysis']['previous_year_samples'] = len(pre_regime_data)
                selection_info['regime_analysis']['regime_post_break_samples'] = len(post_regime_data)
                selection_info['regime_analysis']['previous_year_peak_months'] = pre_regime_patterns[0]
                selection_info['regime_analysis']['regime_post_break_peak_months'] = post_regime_patterns[0]
            else:
                # fallback
                previous_year = hist_with_dates[hist_with_dates['Year'] < regime_break_year]
                regime_post_break = hist_with_dates[hist_with_dates['Year'] >= regime_break_year]
            
            # Monthly aggregation for seasonality analysis
            monthly_hist = hist_with_dates.groupby(['Year', 'Month'])[demand_col].sum().reset_index()
            monthly_means = monthly_hist.groupby('Month')[demand_col].mean()
            monthly_nonzero = monthly_hist[monthly_hist[demand_col] > 0].groupby('Month').size()
            
            # DYNAMIC REGIME-SPECIFIC SEASONALITY ANALYSIS
            if len(previous_year) > 0:
                previous_year_monthly = previous_year.groupby(['Year', 'Month'])[demand_col].sum().reset_index()
                previous_year_peak_months = previous_year_monthly[previous_year_monthly[demand_col] > 0].groupby('Month').size()
                previous_year_peaks = previous_year_peak_months[previous_year_peak_months >= 1].index.tolist()
                # Update with dynamic results
                if 'pre_regime_peaks' in selection_info['regime_analysis']:
                    previous_year_peaks = selection_info['regime_analysis']['pre_regime_peaks']
                selection_info['regime_analysis']['previous_year_peak_months'] = previous_year_peaks
                
            if len(regime_post_break) > 0:
                regime_post_break_monthly = regime_post_break.groupby(['Year', 'Month'])[demand_col].sum().reset_index()
                regime_post_break_peak_months = regime_post_break_monthly[regime_post_break_monthly[demand_col] > 0].groupby('Month').size()
                regime_post_break_peaks = regime_post_break_peak_months[regime_post_break_peak_months >= 1].index.tolist()
                if 'post_regime_peaks' in selection_info['regime_analysis']:
                    regime_post_break_peaks = selection_info['regime_analysis']['post_regime_peaks']
                selection_info['regime_analysis']['regime_post_break_peak_months'] = regime_post_break_peaks
                
                # DYNAMIC PATTERN DETECTION
                bi_annual_pattern = False
                quarterly_pattern = False
                detected_pattern_months = []
                
                if len(regime_post_break_peaks) == 2:
                    gap = abs(regime_post_break_peaks[1] - regime_post_break_peaks[0])
                    if gap in [2, 3, 4, 5, 6, 7]:  # 2-7 months apart (flexible bi-annual/quarterly)
                        bi_annual_pattern = True
                        detected_pattern_months = regime_post_break_peaks
                    elif gap == 9:
                        quarterly_pattern = True
                        detected_pattern_months = regime_post_break_peaks
                
                jul_oct_pattern = any(month in [7, 10] for month in regime_post_break_peaks)
                mar_jun_pattern = any(month in [3, 6] for month in regime_post_break_peaks)
                
                # Store dynamic and fix pattern
                selection_info['regime_analysis']['bi_annual_pattern'] = bi_annual_pattern
                selection_info['regime_analysis']['quarterly_pattern'] = quarterly_pattern
                selection_info['regime_analysis']['detected_pattern_months'] = detected_pattern_months
                selection_info['regime_analysis']['jul_oct_pattern'] = jul_oct_pattern
                selection_info['regime_analysis']['mar_jun_pattern'] = mar_jun_pattern
            
            # DYNAMIC CONSTRAINT ANALYSIS
            total_orders = len(hist_with_dates[hist_with_dates[demand_col] > 0])
            
            pre_regime_primary = selection_info['regime_analysis'].get('pre_regime_primary_size', 25)  # fallback to 25
            post_regime_primary = selection_info['regime_analysis'].get('post_regime_primary_size', 50)  # fallback to 50
            
            # Count occurrences of primary order sizes
            orders_by_size = hist_with_dates[hist_with_dates[demand_col] > 0][demand_col].value_counts().sort_index()
            pre_primary_count = orders_by_size.get(pre_regime_primary, 0)
            post_primary_count = orders_by_size.get(post_regime_primary, 0)
            
            min_significance_threshold = max(1, total_orders * 0.025)  # At least 2% or 1 order
            significant_orders = orders_by_size[orders_by_size >= min_significance_threshold].sort_values(ascending=False)

            # top 2 significant order sizes
            if len(significant_orders) >= 2:
                primary_order_size = significant_orders.index[0]
                secondary_order_size = significant_orders.index[1]
            elif len(significant_orders) == 1:
                primary_order_size = significant_orders.index[0]
                secondary_order_size = 50 if primary_order_size != 50 else 25  # Fallback
            else:
                primary_order_size, secondary_order_size = 25, 50  # Full fallback

            primary_order_count = orders_by_size.get(primary_order_size, 0)
            secondary_order_count = orders_by_size.get(secondary_order_size, 0)
            
            selection_info['regime_analysis']['primary_order_size'] = primary_order_size
            selection_info['regime_analysis']['secondary_order_size'] = secondary_order_size
            selection_info['regime_analysis']['primary_order_count'] = primary_order_count
            selection_info['regime_analysis']['secondary_order_count'] = secondary_order_count
            
            # fallback
            order_25_count = primary_order_count if primary_order_size == 25 else secondary_order_count if secondary_order_size == 25 else 0
            order_50_count = primary_order_count if primary_order_size == 50 else secondary_order_count if secondary_order_size == 50 else 0
            
            # Dynamic order size percentages
            selection_info['regime_analysis']['pre_regime_primary_pct'] = (pre_primary_count / total_orders * 100) if total_orders > 0 else 0
            selection_info['regime_analysis']['post_regime_primary_pct'] = (post_primary_count / total_orders * 100) if total_orders > 0 else 0
            
            # Dynamic significant order size percentages
            selection_info['regime_analysis']['primary_order_pct'] = (primary_order_count / total_orders * 100) if total_orders > 0 else 0
            selection_info['regime_analysis']['secondary_order_pct'] = (secondary_order_count / total_orders * 100) if total_orders > 0 else 0
            
            # Legacy percentages for backward compatibility (calculated from dynamic values)
            selection_info['regime_analysis']['order_25_pct'] = (order_25_count / total_orders * 100) if total_orders > 0 else 0
            selection_info['regime_analysis']['order_50_pct'] = (order_50_count / total_orders * 100) if total_orders > 0 else 0
            
            # Seasonality strength metrics
            cv_monthly = monthly_means.std() / (monthly_means.mean() + 1e-6)  # Coefficient of variation
            seasonal_consistency = len(monthly_nonzero) / 12  # % of months with demand
            peak_months = monthly_nonzero[monthly_nonzero >= 2].index.tolist()  # Months with repeated demand
            
            selection_info['reasoning'].append(f"Seasonality: CV={cv_monthly:.2f}, Consistency={seasonal_consistency:.2f}")
            selection_info['reasoning'].append(f"Peak months: {peak_months}")
            
            # DYNAMIC REGIME-AWARE INSIGHTS
            if regime_break_year and regime_confidence > 0.15:
                selection_info['reasoning'].append(f"Dynamic regime break detected: {regime_break_year} (confidence: {regime_confidence:.2f})")
                selection_info['reasoning'].append(f"Pre-regime: {len(pre_regime_data)} samples, Post-regime: {len(post_regime_data)} samples")
                
                # Dynamic pattern insights
                if selection_info['regime_analysis'].get('bi_annual_pattern', False):
                    pattern_months = selection_info['regime_analysis'].get('detected_pattern_months', [])
                    selection_info['reasoning'].append(f"Bi-annual pattern detected in months: {pattern_months}")
                elif selection_info['regime_analysis'].get('quarterly_pattern', False):
                    pattern_months = selection_info['regime_analysis'].get('detected_pattern_months', [])
                    selection_info['reasoning'].append(f"Quarterly pattern detected in months: {pattern_months}")
                    
                # Dynamic order size info
                if pre_regime_primary != post_regime_primary:
                    selection_info['reasoning'].append(f"Order size evolution: {pre_regime_primary}→{post_regime_primary} units")
                    selection_info['reasoning'].append(f"Primary sizes: Pre-regime {pre_regime_primary} ({selection_info['regime_analysis']['pre_regime_primary_pct']:.1f}%), Post-regime {post_regime_primary} ({selection_info['regime_analysis']['post_regime_primary_pct']:.1f}%)")
            else:
                # fallback
                if len(regime_post_break) > 0:
                    selection_info['reasoning'].append(f"Legacy 2024 regime detected: {len(regime_post_break)} samples, peaks in months {regime_post_break_peaks}")
                    if jul_oct_pattern:
                        selection_info['reasoning'].append(f"Jul/Oct pattern confirmed (not Mar/Jun)")
                        
            if primary_order_count > 0 or secondary_order_count > 0:
                selection_info['reasoning'].append(f"Significant order patterns: {primary_order_size}-unit ({primary_order_count}, {selection_info['regime_analysis']['primary_order_pct']:.1f}%), {secondary_order_size}-unit ({secondary_order_count}, {selection_info['regime_analysis']['secondary_order_pct']:.1f}%)")
                # legacy for reference if different from dynamic
                if (primary_order_size != 25 and secondary_order_size != 25 and order_25_count > 0) or (primary_order_size != 50 and secondary_order_size != 50 and order_50_count > 0):
                    selection_info['reasoning'].append(f"Legacy patterns: 25-unit ({order_25_count}, {selection_info['regime_analysis']['order_25_pct']:.1f}%), 50-unit ({order_50_count}, {selection_info['regime_analysis']['order_50_pct']:.1f}%)")
            
            # Structural break analysis
            structural_break_detected = False
            post_break_regime_different = False
            
            if break_info_result and len(break_info_result) >= 2:
                break_date = break_info_result[0] if break_info_result[0] else None
                break_strength = break_info_result[1] if break_info_result[1] else 0
                
                if hasattr(break_date, '__len__') and not isinstance(break_date, str):
                    break_date = break_date[0] if len(break_date) > 0 else None
                
                if break_date and break_strength > 0.3:  # Strong structural break
                    structural_break_detected = True
                    selection_info['reasoning'].append(f"Structural break detected: {break_date}, strength={break_strength:.2f}")
                    
                    # Analyze pre vs post break patterns
                    try:
                        break_datetime = pd.to_datetime(break_date)
                        pre_break = hist_with_dates[hist_with_dates['PERFORM_DATE'] < break_datetime][demand_col]
                        post_break = hist_with_dates[hist_with_dates['PERFORM_DATE'] >= break_datetime][demand_col]
                        
                        if len(post_break) >= 6:
                            pre_mean = pre_break.mean()
                            post_mean = post_break.mean()
                            regime_change = abs(post_mean - pre_mean) / (pre_mean + 1e-6)
                            
                            if regime_change > 0.5:
                                post_break_regime_different = True
                                selection_info['reasoning'].append(f"Regime change: {pre_mean:.1f} → {post_mean:.1f} (change: {regime_change:.1%})")
                    except Exception as e:
                        selection_info['reasoning'].append(f"Break analysis failed: {str(e)}")
            
            
            regime_break_detected = regime_break_year and regime_confidence > 0.3
            post_regime_dominant = len(post_regime_data) >= 6 if regime_break_detected else len(regime_post_break) >= 6
            regime_transition_detected = len(pre_regime_data) > 0 and len(post_regime_data) > 0 if regime_break_detected else len(previous_year) > 0 and len(regime_post_break) > 0
            
            # Use dynamic patterns or fallback to legacy
            dynamic_bi_annual = selection_info['regime_analysis'].get('bi_annual_pattern', False)
            dynamic_quarterly = selection_info['regime_analysis'].get('quarterly_pattern', False)
            dynamic_post_regime_primary = selection_info['regime_analysis'].get('post_regime_primary_size', 50)
            dynamic_pre_regime_primary = selection_info['regime_analysis'].get('pre_regime_primary_size', 25)
            
            jul_oct_pattern = selection_info['regime_analysis'].get('jul_oct_pattern', False)
            
            # Calculate overall regime strength using dynamic metrics
            regime_strength_base = 0.0
            if regime_break_detected:
                regime_strength_base += 0.5  # Higher base strength for detected break
            elif post_regime_dominant:
                regime_strength_base += 0.4
                
            if dynamic_bi_annual:
                regime_strength_base += 0.3
            elif jul_oct_pattern:
                regime_strength_base += 0.25
                
            if dynamic_post_regime_primary != dynamic_pre_regime_primary:
                order_evolution_strength = min(0.3, selection_info['regime_analysis'].get('post_regime_primary_pct', 0) / 100 * 0.3)
                regime_strength_base += order_evolution_strength
            
            selection_info['regime_analysis']['regime_strength'] = min(1.0, regime_strength_base)
            
            # DYNAMIC SELECTION LOGIC
            if regime_break_detected and dynamic_bi_annual:
                # Dynamically detected regime with bi-annual pattern
                selection_info['method'] = 'dynamic_regime_bi_annual'
                selection_info['weights'] = {'mean': 0.1, 'naive': 0.2, 'seasonal': 0.65, 'drift': 0.05}
                pattern_months = selection_info['regime_analysis'].get('detected_pattern_months', [])
                selection_info['reasoning'].append(f"Selected: Dynamic regime bi-annual (detected pattern in months {pattern_months})")
                
            elif regime_break_detected and dynamic_quarterly:
                # Dynamically detected regime with quarterly pattern
                selection_info['method'] = 'dynamic_regime_quarterly' 
                selection_info['weights'] = {'mean': 0.15, 'naive': 0.25, 'seasonal': 0.5, 'drift': 0.1}
                pattern_months = selection_info['regime_analysis'].get('detected_pattern_months', [])
                selection_info['reasoning'].append(f"Selected: Dynamic regime quarterly (detected pattern in months {pattern_months})")
                
            elif post_regime_dominant and jul_oct_pattern:
                selection_info['method'] = 'regime_post_break_seasonal'
                selection_info['weights'] = {'mean': 0.15, 'naive': 0.25, 'seasonal': 0.55, 'drift': 0.05}
                selection_info['reasoning'].append("Selected: 2024 regime seasonal (Jul/Oct bi-annual pattern)")
                
            elif regime_transition_detected and dynamic_post_regime_primary != dynamic_pre_regime_primary:
                # Dynamic order size transition detected
                post_regime_samples = len(post_regime_data) if regime_break_detected else len(regime_post_break)
                if post_regime_samples >= 3:
                    selection_info['method'] = 'dynamic_regime_transition'
                    selection_info['weights'] = {'mean': 0.2, 'naive': 0.35, 'seasonal': 0.4, 'drift': 0.05}
                    selection_info['reasoning'].append(f"Selected: Dynamic regime transition ({dynamic_pre_regime_primary}→{dynamic_post_regime_primary} unit evolution)")
                else:
                    selection_info['method'] = 'early_transition'
                    selection_info['weights'] = {'mean': 0.1, 'naive': 0.5, 'seasonal': 0.3, 'drift': 0.1}
                    selection_info['reasoning'].append("Selected: Early transition (recent observations critical)")
                    
            elif regime_transition_detected and (secondary_order_count > 0 or order_50_count > 0):
                # Legacy/fallback transition logic using dynamic order sizes
                transition_indicator = secondary_order_count if secondary_order_count > 0 else order_50_count
                transition_size = secondary_order_size if secondary_order_count > 0 else 50                
                if len(regime_post_break) >= 3:
                    selection_info['method'] = 'regime_transition'
                    selection_info['weights'] = {'mean': 0.2, 'naive': 0.4, 'seasonal': 0.35, 'drift': 0.05}
                    selection_info['reasoning'].append(f"Selected: Regime transition ({primary_order_size}→{transition_size} unit evolution, recent bias)")
                else:
                    # Early transition phase - rely more on recent observations
                    selection_info['method'] = 'early_transition'
                    selection_info['weights'] = {'mean': 0.1, 'naive': 0.5, 'seasonal': 0.3, 'drift': 0.1}
                    selection_info['reasoning'].append("Selected: Early transition (recent observations critical)")
                    
            elif structural_break_detected and post_break_regime_different:
                # Strong structural break with regime change
                if len(post_break) >= 12:  # Sufficient post-break data
                    selection_info['method'] = 'mean'  # Use post-break mean
                    selection_info['weights'] = {'mean': 0.6, 'naive': 0.3, 'seasonal': 0.05, 'drift': 0.05}
                    selection_info['reasoning'].append("Selected: Mean (post-break regime established)")
                else:
                    selection_info['method'] = 'naive'  # Recent observations more reliable
                    selection_info['weights'] = {'mean': 0.1, 'naive': 0.6, 'seasonal': 0.15, 'drift': 0.15}
                    selection_info['reasoning'].append("Selected: Naive (insufficient post-break data)")
                    
            elif cv_monthly > 1.0 and len(peak_months) >= 2:
                # Strong seasonality with consistent peak months
                if 'jul_oct_pattern' in selection_info['regime_analysis'] and selection_info['regime_analysis']['jul_oct_pattern']:
                    selection_info['method'] = 'seasonal_jul_oct'
                    selection_info['weights'] = {'mean': 0.05, 'naive': 0.15, 'seasonal': 0.75, 'drift': 0.05}
                    selection_info['reasoning'].append("Selected: Seasonal Jul/Oct (strong seasonal + regime timing)")
                else:
                    selection_info['method'] = 'seasonal'
                    selection_info['weights'] = {'mean': 0.1, 'naive': 0.1, 'seasonal': 0.7, 'drift': 0.1}
                    selection_info['reasoning'].append("Selected: Seasonal (strong seasonal pattern detected)")
                
            elif seasonal_consistency > 0.3 and cv_monthly > 0.5:
                # Moderate seasonality with regime awareness
                if post_regime_dominant:
                    selection_info['method'] = 'seasonal_weighted_post_regime'
                    selection_info['weights'] = {'mean': 0.15, 'naive': 0.2, 'seasonal': 0.55, 'drift': 0.1}
                    selection_info['reasoning'].append("Selected: Seasonal-weighted post-regime (moderate seasonal + regime bias)")
                else:
                    selection_info['method'] = 'seasonal_weighted'
                    selection_info['weights'] = {'mean': 0.2, 'naive': 0.15, 'seasonal': 0.5, 'drift': 0.15}
                    selection_info['reasoning'].append("Selected: Seasonal-weighted (moderate seasonal pattern)")
                
            elif historical_data.std() / (historical_data.mean() + 1e-6) < 0.5:
                # Low variability - stable demand (with regime adjustment)
                if regime_transition_detected:
                    selection_info['method'] = 'mean_regime_adjusted'
                    selection_info['weights'] = {'mean': 0.45, 'naive': 0.35, 'seasonal': 0.15, 'drift': 0.05}
                    selection_info['reasoning'].append("Selected: Mean regime-adjusted (stable demand + regime transition)")
                else:
                    selection_info['method'] = 'mean'
                    selection_info['weights'] = {'mean': 0.6, 'naive': 0.2, 'seasonal': 0.1, 'drift': 0.1}
                    selection_info['reasoning'].append("Selected: Mean (stable demand pattern)")
                
            else:
                # Default to balanced ensemble
                selection_info['method'] = 'ensemble'
                selection_info['weights'] = {'mean': 0.15, 'naive': 0.25, 'seasonal': 0.35, 'drift': 0.25}
                selection_info['reasoning'].append("Selected: Ensemble (no clear pattern dominance)")
            
            return selection_info
        
        try:
            selection_result = select_best_naive_method(historical_demand, break_info_result)

            def update_dynamic_seasonal_method():
                
                regime_info = selection_result.get('regime_analysis', {})
                
                # Use regime_break_year from selection_result or fallback to current year - 1
                dynamic_break_year = regime_info.get('break_year', datetime.now().year - 1)
                dynamic_post_regime_peaks = regime_info.get('post_regime_peaks', [7, 10])
                dynamic_pre_regime_peaks = regime_info.get('pre_regime_peaks', [3, 6])
                dynamic_post_primary_size = regime_info.get('post_regime_primary_size', 50)
                dynamic_pre_primary_size = regime_info.get('pre_regime_primary_size', 25)
                
                # Update the regime the seasonal forecast with detected patterns
                updated_regime_seasonal = []
                for i, date in enumerate(forecast_dates):
                    try:
                        month = date.month
                        year = date.year
                        is_post_regime_forecast = year >= dynamic_break_year
                        if is_post_regime_forecast and month in dynamic_post_regime_peaks:
                            post_regime_hist = forecast_data[
                                (forecast_data['PERFORM_DATE'].dt.month.isin(dynamic_post_regime_peaks)) &
                                (forecast_data['PERFORM_DATE'].dt.year >= dynamic_break_year)
                            ][demand_col]
                            print(f"[DEBUG] i={i}, post_regime_hist values: {post_regime_hist.values}")
                            if len(post_regime_hist) > 0:
                                regime_value = post_regime_hist.mean()
                                print(f"[DEBUG] i={i}, post_regime_hist.mean()={regime_value}")
                            else:
                                regime_value = seasonal_naive[i]
                                print(f"[DEBUG] i={i}, post_regime_hist empty, using seasonal_naive={regime_value}")
                        elif is_post_regime_forecast and month in dynamic_pre_regime_peaks:
                            pre_regime_hist = forecast_data[
                                (forecast_data['PERFORM_DATE'].dt.month.isin(dynamic_pre_regime_peaks)) &
                                (forecast_data['PERFORM_DATE'].dt.year < dynamic_break_year)
                            ][demand_col]
                            if len(pre_regime_hist) > 0:
                                historical_value = pre_regime_hist.mean()
                                recent_value = y_mean
                                regime_value = 0.3 * historical_value + 0.7 * recent_value
                            else:
                                regime_value = seasonal_naive[i]
                        else:
                            regime_value = seasonal_naive[i]
                        # Apply dynamic constraint with actual detected sizes
                        hist_orders = forecast_data[forecast_data[demand_col] > 0][demand_col]
                        if len(hist_orders) > 0:
                            large_order_threshold = dynamic_post_primary_size * 0.9
                            recent_large_orders = len(hist_orders[hist_orders >= large_order_threshold]) / len(hist_orders)
                            if recent_large_orders > 0.3 and regime_value > 0:
                                if regime_value < dynamic_pre_primary_size:
                                    regime_value = 0
                                elif dynamic_pre_primary_size <= regime_value < (dynamic_post_primary_size + dynamic_pre_primary_size) / 2:
                                    regime_value = dynamic_post_primary_size
                                else:
                                    new_val = round(regime_value / dynamic_post_primary_size) * dynamic_post_primary_size
                                    regime_value = new_val
                        if regime_value <= 0:
                            regime_value = y_mean
                        
                        updated_regime_seasonal.append(max(0, regime_value))
                    except Exception as e:
                        updated_regime_seasonal.append(seasonal_naive[i])

                naive_forecasts['Forecast_Seasonal_Regime'] = updated_regime_seasonal
                
                # Apply bi-annual pattern boost if detected
                if selection_result.get('regime_analysis', {}).get('bi_annual_pattern', False):
                    detected_months = selection_result['regime_analysis'].get('detected_pattern_months', [])
                    
                    for i, date in enumerate(forecast_dates):
                        if date.month in detected_months:
                            original_value = naive_forecasts.loc[naive_forecasts.index[i], 'Forecast_Seasonal_Regime']
                            new_value = original_value * 10
                            naive_forecasts.loc[naive_forecasts.index[i], 'Forecast_Seasonal_Regime'] = new_value
                            
            update_dynamic_seasonal_method()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            selection_result = {
                'method': 'seasonal_dominant',
                'weights': {'mean': 0.05, 'naive': 0.15, 'seasonal': 0.70, 'drift': 0.10},
                'reasoning': [f'Fallback: Smart selection failed, but applying seasonal dominance for sporadic bi-annual item']
            }
        
        weights = selection_result['weights']
        
        # Use regime-aware seasonal method for regime-aware selections
        regime_aware_methods = [
            'dynamic_regime_bi_annual', 'dynamic_regime_quarterly', 'dynamic_regime_transition',
            'regime_post_break_seasonal', 'regime_transition', 'seasonal_jul_oct', 
            'seasonal_weighted_post_regime', 'seasonal_weighted_2024', 'mean_regime_adjusted', 'early_transition'
        ]
        
        if selection_result['method'] in regime_aware_methods:
            # Use regime-aware seasonal method
            seasonal_component = naive_forecasts['Forecast_Seasonal_Regime']
            selection_result['reasoning'].append("Using regime-aware seasonal method for temporal accuracy")
        else:
            # Use standard seasonal method
            seasonal_component = naive_forecasts['Forecast_Seasonal']
        
        naive_forecasts['Forecast_Smart'] = (
            naive_forecasts['Forecast_Mean'] * weights['mean'] + 
            naive_forecasts['Forecast_Naive'] * weights['naive'] + 
            seasonal_component * weights['seasonal'] + 
            naive_forecasts['Forecast_Drift'] * weights['drift']
        )
        
        # Blend standard and regime-aware seasonality based on regime strength
        regime_strength = 0.0
        if 'regime_analysis' in selection_result:
            # Calculate regime strength based on Previous year's data and Jul/Oct patterns
            regime_post_break_samples = selection_result['regime_analysis'].get('regime_post_break_samples', 0)
            jul_oct_pattern = selection_result['regime_analysis'].get('jul_oct_pattern', False)
            primary_order_pct = selection_result['regime_analysis'].get('primary_order_pct', 0)
            secondary_order_pct = selection_result['regime_analysis'].get('secondary_order_pct', 0)
            
            significant_order_impact = max(primary_order_pct, secondary_order_pct)
            
            regime_strength = min(1.0, (regime_post_break_samples / 12) * 0.4 + 
                                        (1.0 if jul_oct_pattern else 0.0) * 0.3 + 
                                        (significant_order_impact / 100) * 0.3)
                    
        naive_forecasts['Forecast_Regime_Blend'] = (
            naive_forecasts['Forecast_Seasonal'] * (1 - regime_strength) +
            naive_forecasts['Forecast_Seasonal_Regime'] * regime_strength
        )
        
        naive_forecasts['Forecast_Enhanced'] = (
            naive_forecasts['Forecast_Mean'] * 0.2 + 
            naive_forecasts['Forecast_Naive'] * 0.25 + 
            naive_forecasts['Forecast_Regime_Blend'] * 0.45 + 
            naive_forecasts['Forecast_Drift'] * 0.1
        )
        
        # Standard ensemble
        naive_forecasts['Forecast_Ensemble'] = (
            naive_forecasts['Forecast_Mean'] + 
            naive_forecasts['Forecast_Naive'] + 
            naive_forecasts['Forecast_Seasonal'] + 
            naive_forecasts['Forecast_Drift']
        ) / 4       
        
        def calculate_seasonal_uncertainty(data, period=12):
            if len(data) < period:
                return data.std()
            
            # Calculate seasonal residuals
            seasonal_residuals = []
            for i in range(period, len(data)):
                seasonal_forecast = data.iloc[i - period]
                actual = data.iloc[i]
                seasonal_residuals.append(abs(actual - seasonal_forecast))
            
            return np.std(seasonal_residuals) if seasonal_residuals else data.std()
        
        def calculate_trend_uncertainty(data, window=6):
            if len(data) < window:
                return data.std()
            
            trend_residuals = []
            for i in range(window, len(data)):
                recent_data = data.iloc[i-window:i]
                if len(recent_data) >= 2:
                    trend_slope = (recent_data.iloc[-1] - recent_data.iloc[0]) / (len(recent_data) - 1)
                    trend_forecast = recent_data.iloc[-1] + trend_slope
                    actual = data.iloc[i]
                    trend_residuals.append(abs(actual - trend_forecast))
            
            return np.std(trend_residuals) if trend_residuals else data.std()
                
        def monte_carlo_ci(regime_analysis, forecast_values, historical_data, n_simulations=500):
            
            simulations = np.zeros((len(forecast_values), n_simulations))
            
            base_std = historical_data.std()
            historical_mean = historical_data.mean()
            
            uncertainty_ratio = min(0.3, base_std / max(1, historical_mean))  # Cap at 30%
            
            for sim in range(n_simulations):
                for i, forecast_point in enumerate(forecast_values):
                    # Scale noise relative to forecast magnitude
                    scaled_std = forecast_point * uncertainty_ratio
                    
                    # Add regime uncertainty (10% chance of higher/lower regime)
                    if np.random.random() < 0.1:
                        regime_multiplier = np.random.uniform(0.7, 1.4)  # ±40% regime shift
                    else:
                        regime_multiplier = 1.0
                    
                    base_forecast = forecast_point * regime_multiplier
                    simulated_value = max(0, np.random.normal(base_forecast, scaled_std))
                    simulations[i, sim] = simulated_value
            
            ci_results = {
                'CI_05': np.percentile(simulations, 5, axis=1),
                'CI_95': np.percentile(simulations, 95, axis=1)
            }
            return ci_results
        
        
        ci_05_methods = []
        ci_95_methods = []
        
        
        blend_mc_success = False
        
        try:
            mc_ci_blend = monte_carlo_ci(selection_result, naive_forecasts['Forecast_Regime_Blend'], historical_demand)
            naive_forecasts['Forecast_Regime_Blend_MonteCarlo_CI_05'] = mc_ci_blend['CI_05']
            naive_forecasts['Forecast_Regime_Blend_MonteCarlo_CI_95'] = mc_ci_blend['CI_95']
            blend_mc_success = True
        except Exception as e:
            print(f"Monte Carlo CI for Regime Blend failed: {str(e)}")
        
        blend_ci_05_methods = []
        blend_ci_95_methods = []
        
        if blend_mc_success:
            blend_ci_05_methods.append(naive_forecasts['Forecast_Regime_Blend_MonteCarlo_CI_05'])
            blend_ci_95_methods.append(naive_forecasts['Forecast_Regime_Blend_MonteCarlo_CI_95'])
                    
        else:
            fallback_std = historical_demand.std()
            naive_forecasts['Forecast_Regime_Blend_Ensemble_CI_05'] = np.maximum(0, naive_forecasts['Forecast_Regime_Blend'] - 1.645 * fallback_std)
            naive_forecasts['Forecast_Regime_Blend_Ensemble_CI_95'] = naive_forecasts['Forecast_Regime_Blend'] + 1.645 * fallback_std
            naive_forecasts['Forecast_Regime_Blend_CI_Width'] = naive_forecasts['Forecast_Regime_Blend_Ensemble_CI_95'] - naive_forecasts['Forecast_Regime_Blend_Ensemble_CI_05']
                
        def apply_seasonal_clustering(naive_forecasts, historical_demand):
            """Apply seasonal pattern clustering to align with historical monthly patterns"""
            
            historical_monthly = historical_demand.groupby(['Year', 'Month']).agg({
                'Forecast_Regime_Blend': 'sum'
            }).reset_index()           
            unique_historical_values = sorted(historical_monthly['Forecast_Regime_Blend'].unique())
                        
            seasonal_clusters = {
                'winter_spring': list(range(1, 7)),
                'summer_fall': list(range(7, 12)),
                'december': [12]
            }
            
            cluster_targets = {}
            for cluster_name, months in seasonal_clusters.items():
                cluster_mask = historical_monthly['Month'].isin(months)
                cluster_values = sorted(historical_monthly[cluster_mask]['Forecast_Regime_Blend'].unique())
                cluster_targets[cluster_name] = cluster_values
            
            # If any cluster has no historical data, use all available values as fallback
            for cluster_name in cluster_targets:
                if not cluster_targets[cluster_name]:
                    cluster_targets[cluster_name] = unique_historical_values
            
            clustered_forecasts = naive_forecasts.copy()
            monthly_groups = clustered_forecasts.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month'])
            
            redistribution_log = []
            
            for (year, month), group in monthly_groups:
                current_monthly_total = group['Forecast_Regime_Blend'].sum()
                
                if current_monthly_total == 0:
                    continue  # Skip zero months
                
                month_cluster = None
                for cluster, months in seasonal_clusters.items():
                    if month in months:
                        month_cluster = cluster
                        break
                
                if month_cluster is None:
                    continue
                
                valid_targets = cluster_targets[month_cluster]
                
                if current_monthly_total not in valid_targets:
                    closest_target = min(valid_targets, key=lambda x: abs(x - current_monthly_total))
                    
                    if closest_target == 0 and current_monthly_total > 0:
                        non_zero_targets = [t for t in valid_targets if t > 0]
                        if non_zero_targets:
                            closest_target = min(non_zero_targets)
                                        
                    if current_monthly_total > 0:
                        scaling_factor = closest_target / current_monthly_total
                    else:
                        scaling_factor = 0
                    
                    # Apply scaling to daily forecasts and CI bounds
                    group_indices = group.index
                    clustered_forecasts.loc[group_indices, 'Forecast_Regime_Blend'] *= scaling_factor
                    clustered_forecasts.loc[group_indices, 'Forecast_Regime_Blend_MonteCarlo_CI_05'] *= scaling_factor
                    clustered_forecasts.loc[group_indices, 'Forecast_Regime_Blend_MonteCarlo_CI_95'] *= scaling_factor
                    
                    redistribution_log.append({
                        'year': year,
                        'month': month,
                        'original': current_monthly_total,
                        'clustered': closest_target,
                        'scaling_factor': scaling_factor,
                        'cluster': month_cluster
                    })
            return clustered_forecasts, redistribution_log
        
        naive_forecasts['Year'] = naive_forecasts['PERFORM_DATE'].dt.year
        naive_forecasts['Month'] = naive_forecasts['PERFORM_DATE'].dt.month
        naive_forecasts['Day'] = naive_forecasts['PERFORM_DATE'].dt.day
              
        # Standard ensemble average of all 4 base methods (for comparison)
        naive_forecasts['Forecast_Ensemble'] = (
            naive_forecasts['Forecast_Mean'] + 
            naive_forecasts['Forecast_Naive'] + 
            naive_forecasts['Forecast_Seasonal'] + 
            naive_forecasts['Forecast_Drift']
        ) / 4
        
        agg_dict = {
            'Forecast_Mean': 'sum',
            'Forecast_Naive': 'sum', 
            'Forecast_Seasonal': 'sum',
            'Forecast_Seasonal_Regime': 'sum',
            'Forecast_Drift': 'sum',
            'Forecast_Ensemble': 'sum',
            'Forecast_Smart': 'sum',
            'Forecast_Enhanced': 'sum',
            'Forecast_Regime_Blend': 'sum'
        }

        ci_columns = [
            'Forecast_Regime_Blend_MonteCarlo_CI_05', 'Forecast_Regime_Blend_MonteCarlo_CI_95',
        ]

        for col in ci_columns:
            if col in naive_forecasts.columns:
                agg_dict[col] = 'sum'

        monthly_forecast = naive_forecasts.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month']).agg(agg_dict).reset_index()
        forecast_months = monthly_forecast['Month'].values[-13:]

        monthly_forecast['Forecast_Regime_Blend_Original'] = monthly_forecast['Forecast_Regime_Blend'].copy()
        
        def apply_demand_consolidation_clustering(naive_forecasts, monthly_forecast):
            """distribute low-demand months into nearby peak months"""
                        
            clustered_forecasts = naive_forecasts.copy()
            redistribution_log = []
            
            # Identify peak months and low months
            monthly_totals = []
            for idx, row in monthly_forecast.iterrows():
                if row['Forecast_Regime_Blend'] > 0:
                    monthly_totals.append({
                        'idx': idx,
                        'year': row['Year'], 
                        'month': row['Month'],
                        'total': row['Forecast_Regime_Blend']
                    })
            
            monthly_totals.sort(key=lambda x: x['total'], reverse=True)
           
            if monthly_totals:
                max_total = monthly_totals[0]['total']
                threshold = max_total * 0.2  # 20% of max = threshold for "low" months
                
                # Separate into peak and low months
                peak_months = [m for m in monthly_totals if m['total'] >= threshold]
                low_months = [m for m in monthly_totals if m['total'] < threshold]
                                
                # Redistribute each low month to nearest peak month
                for low_month in low_months:
                    if not peak_months:
                        continue
                        
                    low_month_num = low_month['year'] * 12 + low_month['month']                    
                    nearest_peak = min(peak_months, 
                                     key=lambda p: abs((p['year'] * 12 + p['month']) - low_month_num))
                    
                    # Transfer daily demand from low month to peak month
                    low_mask = (clustered_forecasts['Year'] == low_month['year']) & (clustered_forecasts['Month'] == low_month['month'])
                    peak_mask = (clustered_forecasts['Year'] == nearest_peak['year']) & (clustered_forecasts['Month'] == nearest_peak['month'])
                    
                    daily_regime_blend = clustered_forecasts.loc[low_mask, 'Forecast_Regime_Blend'].values
                    daily_ci_05 = clustered_forecasts.loc[low_mask, 'Forecast_Regime_Blend_MonteCarlo_CI_05'].values if 'Forecast_Regime_Blend_MonteCarlo_CI_05' in clustered_forecasts.columns else None
                    daily_ci_95 = clustered_forecasts.loc[low_mask, 'Forecast_Regime_Blend_MonteCarlo_CI_95'].values if 'Forecast_Regime_Blend_MonteCarlo_CI_95' in clustered_forecasts.columns else None
                    
                    # Add to peak months
                    peak_days = clustered_forecasts.loc[peak_mask].shape[0]
                    if peak_days > 0:
                        daily_addition = daily_regime_blend.sum() / peak_days
                        clustered_forecasts.loc[peak_mask, 'Forecast_Regime_Blend'] += daily_addition
                        
                        # Transfer CI bounds proportionally
                        if daily_ci_05 is not None and daily_ci_95 is not None:
                            ci_05_addition = daily_ci_05.sum() / peak_days
                            ci_95_addition = daily_ci_95.sum() / peak_days
                            clustered_forecasts.loc[peak_mask, 'Forecast_Regime_Blend_MonteCarlo_CI_05'] += ci_05_addition
                            clustered_forecasts.loc[peak_mask, 'Forecast_Regime_Blend_MonteCarlo_CI_95'] += ci_95_addition
                    
                    # Low months to zero
                    clustered_forecasts.loc[low_mask, 'Forecast_Regime_Blend'] = 0
                    if daily_ci_05 is not None:
                        clustered_forecasts.loc[low_mask, 'Forecast_Regime_Blend_MonteCarlo_CI_05'] = 0
                        clustered_forecasts.loc[low_mask, 'Forecast_Regime_Blend_MonteCarlo_CI_95'] = 0
                    
                    # Update peak month total for next iteration
                    for peak in peak_months:
                        if peak['year'] == nearest_peak['year'] and peak['month'] == nearest_peak['month']:
                            peak['total'] += low_month['total']
                            break
                    
                    redistribution_log.append({
                        'from_year': low_month['year'],
                        'from_month': low_month['month'], 
                        'to_year': nearest_peak['year'],
                        'to_month': nearest_peak['month'],
                        'amount': low_month['total']
                    })
            
            return clustered_forecasts, redistribution_log
        
        naive_forecasts, clustering_log = apply_demand_consolidation_clustering(naive_forecasts, monthly_forecast)
        
        monthly_forecast = naive_forecasts.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month']).agg(agg_dict).reset_index()
        
        if clustering_log and len(clustering_log) > 0 and 'year' in clustering_log[0]:
            monthly_forecast['Forecast_Regime_Blend_Original'] = monthly_forecast['Forecast_Regime_Blend'].copy()
            # Restore original values where clustering occurred
            for entry in clustering_log:
                year_mask = (monthly_forecast['Year'] == entry['year'])
                month_mask = (monthly_forecast['Month'] == entry['month'])
                combined_mask = year_mask & month_mask
                monthly_forecast.loc[combined_mask, 'Forecast_Regime_Blend_Original'] = entry['original']
        
        
        forecast_columns = ['Forecast_Mean', 'Forecast_Naive', 'Forecast_Seasonal', 
                          'Forecast_Seasonal_Regime', 'Forecast_Drift', 'Forecast_Ensemble', 
                          'Forecast_Smart', 'Forecast_Enhanced', 'Forecast_Regime_Blend']
        
        discretizable_columns = forecast_columns.copy()
        ci_forecast_columns = [col for col in ci_columns if col in monthly_forecast.columns]
        discretizable_columns.extend(ci_forecast_columns)
        
        for col in discretizable_columns:
                discretized_col = f"{col}_Discretized"
                
                monthly_values = list(monthly_forecast[col].values)
                peak_demand = max(monthly_values)
                primary_threshold = peak_demand * 0.05    # 5% for noise elimination
                
                if 'bi_annual_pattern' in selection_result.get('regime_analysis', {}) and selection_result['regime_analysis'].get('bi_annual_pattern', False):
                    secondary_threshold = peak_demand * 0.98  # 98% for bi-annual - keep only very top months
                    bi_annual_discretization = True
                else:
                    secondary_threshold = peak_demand * 0.20  # 20% to preserve secondary signals like 7.96
                    bi_annual_discretization = False
                
                if col == 'Forecast_Smart':
                    threshold_type = "BI-ANNUAL (98%)" if bi_annual_discretization else "REGULAR (20%)"
                
                non_zero_values = [v for v in monthly_values if v > 0]
                if non_zero_values:
                    sorted_non_zero = sorted(non_zero_values)
                    median_demand = sorted_non_zero[len(sorted_non_zero)//2]
                else:
                    median_demand = 0
                
                discretized_values = []
                values_kept = 0
                values_eliminated = 0
                
                for value in monthly_values:
                    if value < primary_threshold:
                        discretized_values.append(0)  # Below 5% of peak → noise
                        values_eliminated += 1
                    elif value >= secondary_threshold:
                        discretized_values.append(value)  # Above 20% of peak → meaningful
                        values_kept += 1
                    else:
                        # Between 5-20% of peak → evaluate against median
                        if value > median_demand * 2:
                            discretized_values.append(value)  # Keep secondary signals
                            values_kept += 1
                        else:
                            discretized_values.append(0)  # Treat as noise
                            values_eliminated += 1
                
                monthly_forecast[discretized_col] = discretized_values

        
        cols_to_keep = ['ITEM_NUMBER_SITE', 'Year', 'Month', 'Day', 'PERFORM_DATE', 'Forecast_Regime_Blend', 'Forecast_Regime_Blend_MonteCarlo_CI_05', 'Forecast_Regime_Blend_MonteCarlo_CI_95']
        naive_forecasts = naive_forecasts[cols_to_keep].copy()
        naive_forecasts.rename(columns={'Forecast_Regime_Blend': demand_col, 'Forecast_Regime_Blend_MonteCarlo_CI_05': 'Forecasted Demand Inf', 'Forecast_Regime_Blend_MonteCarlo_CI_95': 'Forecasted Demand Sup'}, inplace=True)
        cols_to_keep = ['ITEM_NUMBER_SITE', 'Year', 'Month', 'Forecast_Regime_Blend', 'Forecast_Regime_Blend_MonteCarlo_CI_05', 'Forecast_Regime_Blend_MonteCarlo_CI_95']
        monthly_forecast = monthly_forecast[cols_to_keep].copy()
        monthly_forecast.rename(columns={'Forecast_Regime_Blend': demand_col, 'Forecast_Regime_Blend_MonteCarlo_CI_05': 'Forecasted Demand Inf', 'Forecast_Regime_Blend_MonteCarlo_CI_95': 'Forecasted Demand Sup'}, inplace=True)     
        monthly_forecast[demand_col] = monthly_forecast[demand_col].round(0).astype(int)
        monthly_forecast['Forecasted Demand Inf'] = monthly_forecast['Forecasted Demand Inf'].round(0).astype(int)
        monthly_forecast['Forecasted Demand Sup'] = monthly_forecast['Forecasted Demand Sup'].round(0).astype(int)
        
        def concentrate_monthly_to_daily(daily_forecasts, monthly_totals, concentration_strategy='mid_month'):
            naive_forecasts[demand_col] = 0
            naive_forecasts['Forecasted Demand Inf'] = 0
            naive_forecasts['Forecasted Demand Sup'] = 0
            
            for _, month_row in monthly_totals.iterrows():
                year = month_row['Year']
                month = month_row['Month']
                monthly_total = month_row[demand_col]
                monthly_inf = month_row['Forecasted Demand Inf'] 
                monthly_sup = month_row['Forecasted Demand Sup']
                
                if monthly_total > 0:
                    month_mask = (naive_forecasts['Year'] == year) & (naive_forecasts['Month'] == month)
                    days_in_month = naive_forecasts[month_mask]
                    
                    if len(days_in_month) > 0:
                        if concentration_strategy == 'mid_month':
                            target_day = 15
                            available_days = days_in_month['Day'].values
                            chosen_day = min(available_days, key=lambda x: abs(x - target_day))
                            day_mask = month_mask & (naive_forecasts['Day'] == chosen_day)
                            
                        elif concentration_strategy == 'random_deterministic':
                            np.random.seed(year * 100 + month)
                            chosen_day_idx = np.random.choice(len(days_in_month))
                            chosen_day = days_in_month.iloc[chosen_day_idx]['Day']
                            day_mask = month_mask & (naive_forecasts['Day'] == chosen_day)
                            
                        elif concentration_strategy == 'business_days':
                            if monthly_total > 0:
                                target_day = 1 if month % 2 == 1 else 15
                                available_days = days_in_month['Day'].values
                                chosen_day = min(available_days, key=lambda x: abs(x - target_day))
                                day_mask = month_mask & (naive_forecasts['Day'] == chosen_day)
                        
                        naive_forecasts.loc[day_mask, demand_col] = monthly_total
                        naive_forecasts.loc[day_mask, 'Forecasted Demand Inf'] = monthly_inf
                        naive_forecasts.loc[day_mask, 'Forecasted Demand Sup'] = monthly_sup
                        
                        chosen_date = naive_forecasts.loc[day_mask, 'PERFORM_DATE'].iloc[0]
            
            return naive_forecasts
        
        naive_forecasts = concentrate_monthly_to_daily(naive_forecasts, monthly_forecast, concentration_strategy='mid_month')
        naive_forecasts.drop(columns={'Day'}, inplace=True)
        naive_forecasts['Week'] = ((naive_forecasts['PERFORM_DATE'] - naive_forecasts['PERFORM_DATE'].dt.to_period('M').dt.start_time).dt.days // 7) + 1
        naive_forecasts['History'] = 1
        naive_forecasts['EffectiveNbDays'] = 1
        naive_forecasts['EffectiveDemand'] = np.where(naive_forecasts[demand_col]>0, 1, 0)
        naive_forecasts['Week2'] = np.nan
        if np.issubdtype(naive_forecasts['Week2'].dtype, np.floating):
            naive_forecasts['Week2'] = naive_forecasts['Week2'].astype('Int64')
        naive_forecasts['DaysInMonth'] = naive_forecasts['PERFORM_DATE'].dt.days_in_month

        global_interval = simulate_aggregate_intervals(naive_forecasts, dist='nbinom')
        global_interval['ITEM_NUMBER_SITE'] = item_id
        forecast_data = pd.concat([forecast_data, naive_forecasts], axis=0)
        
        forecast_results_summary = forecast_data.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month']).agg(
            Nb_Days_sum=('EffectiveDemand', 'sum'),
            Forecasted_Demand_sum=(demand_col, 'sum'),
            Forecasted_Demand_mean=(demand_col, 'mean'),
            Forecasted_Demand_std=(demand_col, 'std'),
            Forecasted_Demand_inf_sum=('Forecasted Demand Inf', 'sum'),
            Forecasted_Demand_inf_mean=('Forecasted Demand Inf', 'mean'),
            Forecasted_Demand_inf_std=('Forecasted Demand Inf', 'std'),
            Forecasted_Demand_sup_sum=('Forecasted Demand Sup', 'sum'),
            Forecasted_Demand_sup_mean=('Forecasted Demand Sup', 'mean'),
            Forecasted_Demand_sup_std=('Forecasted Demand Sup', 'std'),
        ).reset_index()
    
        forecast_results_summary.rename(columns={
            'Nb_Days_sum': 'Number of Days of Demand',
            'Forecasted_Demand_sum': 'Forecasted Demand',
            'Forecasted_Demand_mean': 'Forecasted Demand mean',
            'Forecasted_Demand_std': 'Forecasted Demand std',
            'Forecasted_Demand_inf_sum': 'Forecasted Demand Inf',
            'Forecasted_Demand_inf_mean': 'Forecasted Demand Inf mean',
            'Forecasted_Demand_inf_std': 'Forecasted Demand Inf std',
            'Forecasted_Demand_sup_sum': 'Forecasted Demand Sup',
            'Forecasted_Demand_sup_mean': 'Forecasted Demand Sup mean',
            'Forecasted_Demand_sup_std': 'Forecasted Demand Sup std',
        }, inplace=True)
    
        forecast_results_summary['Forecasted Demand mean'] = forecast_results_summary['Forecasted Demand mean'].round(3)
        forecast_results_summary['Forecasted Demand std'] = forecast_results_summary['Forecasted Demand std'].round(3)
        forecast_results_summary['Forecasted Demand Inf mean'] = forecast_results_summary['Forecasted Demand Inf mean'].round(3)
        forecast_results_summary['Forecasted Demand Inf std'] = forecast_results_summary['Forecasted Demand Inf std'].round(3)
        forecast_results_summary['Forecasted Demand Sup mean'] = forecast_results_summary['Forecasted Demand Sup mean'].round(3)
        forecast_results_summary['Forecasted Demand Sup std'] = forecast_results_summary['Forecasted Demand Sup std'].round(3)
        duration = (time.time() - start_time)/60

    if item_selection_bi_annual.loc[item_selection_bi_annual['ITEM_NUMBER_SITE'] == item_id, 
                          'PURCHASE_CYCLE_CATEGORY'].iloc[0] in ["Bi-annual"]:    
        all_global_interval.append(global_interval)
        all_forecasts.append(forecast_data)
        all_summaries.append(forecast_results_summary)
        all_last_years.append(naive_forecasts)

#final_feats = pd.concat(all_best_feats, ignore_index=True)
final_forecast_bi_annual = pd.concat(all_forecasts, ignore_index=True)
monthly_grouped_bi_annual = final_forecast_bi_annual.groupby(['ITEM_NUMBER_SITE', 'Year', 'Month']).agg({'ACTUAL_DEMAND': 'sum',
                                                                                    'EffectiveDemand': 'sum'}).reset_index()

final_summary_bi_annual = pd.concat(all_summaries, ignore_index=True)
final_intervals_bi_annual = pd.concat(all_global_interval, ignore_index=True)
final_last_years_bi_annual = pd.concat(all_last_years, ignore_index=True)
final_forecast_bi_annual.drop(columns={'DaysInMonth', 'EffectiveNbDays', 'EffectiveDemand', 'Week2', 'History'}, inplace=True)



#del final_monthly_summary

export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_summaries_bi_annual.csv'
final_summary_bi_annual.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_forecast_data_bi_annual.csv'
final_forecast_bi_annual.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_intervals_bi_annual.csv'
final_intervals_bi_annual.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')
export_path = fr'G:\My Drive\BIG_FILES\Inventory_optimization\hendrickson\{division}\Datas_out\Forecasts\all_monthly_summaries_bi_annual.csv'
monthly_grouped_bi_annual.to_csv(export_path, index=None, doublequote=False, header=True, sep=";", encoding='UTF-8')

          


   



