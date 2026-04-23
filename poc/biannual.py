"""
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

          
"""