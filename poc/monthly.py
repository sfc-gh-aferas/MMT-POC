###############  Monthly
    """    
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
    """