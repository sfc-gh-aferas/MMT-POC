"""
Monthly tsfresh feature lists and configuration utilities for demand forecasting.

Extracted from original Actual_Demand_Forecast-monotonic - V8 NEW - Cupy.py (lines 3176-3610).
These are carefully curated, hand-selected tsfresh features for optimal forecasting performance.
"""

def get_features_curr_M_list():
    """Get current month tsfresh feature list (originally ~158 features)"""
    return [
        'Actual Demand__index_mass_quantile__q_0.9',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2"',
        'Actual Demand__energy_ratio_by_chunks__num_segments_10__segment_focus_9',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6"',
        'Actual Demand__last_location_of_minimum',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__index_mass_quantile__q_0.8',
        'Actual Demand__agg_autocorrelation__f_agg_"var"__maxlag_40"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_3"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_9"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_1"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0"',
        'Actual Demand__mean_change',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_2"',
        'Actual Demand__mean_second_derivative_central',
        'Actual Demand__index_mass_quantile__q_0.7',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__linear_trend__attr_"rvalue"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_4"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_7"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_6"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_8"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_15"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_5"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"var"',
        'Actual Demand__index_mass_quantile__q_0.6',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_14"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_10"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_9"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_5"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_16"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_11"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_13"',
        'Actual Demand__mean',
        'Actual Demand__mean_n_absolute_max__number_of_maxima_7',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_8"',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_0"',
        'Actual Demand__sum_values',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_0"',
        'Actual Demand__cid_ce__normalize_True',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_12"',
        'Actual Demand__count_below__t_0',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_14"',
        'Actual Demand__count_above_mean',
        'Actual Demand__cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_6"',
        'Actual Demand__quantile__q_0.9',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_16"',
        'Actual Demand__fft_aggregated__aggtype_"skew"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_13"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_12"',
        'Actual Demand__cwt_coefficients__coeff_10__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__binned_entropy__max_bins_10',
        'Actual Demand__last_location_of_maximum',
        'Actual Demand__root_mean_square',
        'Actual Demand__cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_11__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__abs_energy',
        'Actual Demand__first_location_of_maximum',
        'Actual Demand__lempel_ziv_complexity__bins_100',
        'Actual Demand__ratio_beyond_r_sigma__r_1',
        'Actual Demand__agg_autocorrelation__f_agg_"mean"__maxlag_40"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_13"',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_9"',
        'Actual Demand__linear_trend__attr_"slope"',
        'Actual Demand__linear_trend__attr_"intercept"',
        'Actual Demand__lempel_ziv_complexity__bins_10',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_9"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_12"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_4"',
        'Actual Demand__ratio_beyond_r_sigma__r_1.5',
        'Actual Demand__variance',
        'Actual Demand__index_mass_quantile__q_0.4',
        'Actual Demand__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__ar_coefficient__coeff_0__k_10',
        'Actual Demand__fft_aggregated__aggtype_"kurtosis"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_8"',
        'Actual Demand__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)',
        'Actual Demand__standard_deviation',
        'Actual Demand__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_13__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_6__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_3"',
        'Actual Demand__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_aggregated__aggtype_"centroid"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"',
        'Actual Demand__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_11"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_5"',
        'Actual Demand__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_4"',
        'Actual Demand__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_11"',
        'Actual Demand__symmetry_looking__r_0.15000000000000002',
        'Actual Demand__agg_autocorrelation__f_agg_"median"__maxlag_40"',
        'Actual Demand__number_peaks__n_1',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_14"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_16"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_15"',
        'Actual Demand__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__lempel_ziv_complexity__bins_3',
        'Actual Demand__ar_coefficient__coeff_3__k_10',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_7"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_6"',
        'Actual Demand__first_location_of_minimum',
        'Actual Demand__large_standard_deviation__r_0.30000000000000004',
        'Actual Demand__ar_coefficient__coeff_1__k_10',
        'Actual Demand__ar_coefficient__coeff_4__k_10',
        'Actual Demand__lempel_ziv_complexity__bins_5',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_8"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"var"',
        'Actual Demand__ratio_value_number_to_time_series_length',
        'Actual Demand__lempel_ziv_complexity__bins_2',
        'Actual Demand__symmetry_looking__r_0.1',
        'Actual Demand__large_standard_deviation__r_0.25',
        'Actual Demand__ratio_beyond_r_sigma__r_2',
        'Actual Demand__number_cwt_peaks__n_5',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_27"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_24"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_23"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_22"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_19"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_18"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_17"',
        'Actual Demand__energy_ratio_by_chunks__num_segments_10__segment_focus_6',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_7"',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_14"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_27"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_25"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_24"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_22"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_17"',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_15"',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_1"',
        'Actual Demand__fft_coefficient__attr_"abs"__coeff_2"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_10"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_25"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_3"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_31"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_30"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_29"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_28"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_26"',
        'Actual Demand__variation_coefficient'
    ]

def get_features_6_M_list():
    """Get 6-month tsfresh feature list (originally ~86 features)"""
    return [
        'Actual Demand__last_location_of_minimum',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_27"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
        'Actual Demand__energy_ratio_by_chunks__num_segments_10__segment_focus_9',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_52"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_80"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_27"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_53"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_80"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_54"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_26"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_80"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0"',
        'Actual Demand__mean_change',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_54"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_28"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_56"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_55"',
        'Actual Demand__mean_second_derivative_central',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_51"',
        'Actual Demand__index_mass_quantile__q_0.9',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_52"',
        'Actual Demand__cwt_coefficients__coeff_11__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_12__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"',
        'Actual Demand__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_77"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_78"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"',
        'Actual Demand__agg_autocorrelation__f_agg_"var"__maxlag_40"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_24"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_72"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_48"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_71"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_48"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_47"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_53"',
        'Actual Demand__fft_aggregated__aggtype_"skew"',
        'Actual Demand__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_73"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_47"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_46"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_50"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_49"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_49"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_45"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_73"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"',
        'Actual Demand__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_49"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
        'Actual Demand__index_mass_quantile__q_0.8',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_98"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_29"',
        'Actual Demand__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_98"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_97"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_97"',
        'Actual Demand__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_95"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_94"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_6"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_57"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_23"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_96"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_95"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_94"',
        'Actual Demand__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_74"'
    ]

def get_features_12_M_list():
    """Get 12-month tsfresh feature list (originally ~53 features)"""
    return [
        'Actual Demand__last_location_of_minimum',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_53"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_52"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8"',
        'Actual Demand__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6"',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
        'Actual Demand__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
        'Actual Demand__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
        'Actual Demand__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__mean_change',
        'Actual Demand__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0"',
        'Actual Demand__fft_aggregated__aggtype_"skew"',
        'Actual Demand__cwt_coefficients__coeff_12__w_2__widths_(2, 5, 10, 20)',
        'Actual Demand__mean_second_derivative_central',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_54"',
        'Actual Demand__energy_ratio_by_chunks__num_segments_10__segment_focus_9',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"',
        'Actual Demand__agg_autocorrelation__f_agg_"var"__maxlag_40"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_48"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_96"',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_98"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_97"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_95"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_52"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_97"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_94"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_96"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_97"',
        'Actual Demand__fft_coefficient__attr_"angle"__coeff_52"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_95"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_47"',
        'Actual Demand__first_location_of_minimum',
        'Actual Demand__index_mass_quantile__q_0.9',
        'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_98"',
        'Actual Demand__fft_coefficient__attr_"real"__coeff_94"',
        'Actual Demand__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
        'Actual Demand__cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)',  
        'Actual Demand__fft_coefficient__attr_"real"__coeff_49"',
        'Actual Demand__fft_coefficient__attr_"imag"__coeff_93"'
    ]

def parse_feature_name_to_tsfresh_config(feature_name):
    """
    Convert a tsfresh feature name to its configuration parameters.
    Example: 'Actual Demand__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"'
    Returns: ('agg_linear_trend', {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'})
    """
    # Remove the column prefix
    if feature_name.startswith('Actual Demand__'):
        feature_name = feature_name[len('Actual Demand__'):]
    
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
                config[func_name] = []
            
            if params is None:
                # For functions without parameters
                config[func_name] = None
            else:
                # For functions with parameters
                if config[func_name] is None:
                    config[func_name] = [params]
                elif isinstance(config[func_name], list):
                    if params not in config[func_name]:
                        config[func_name].append(params)
                else:
                    config[func_name] = [params]
    
    return config

def get_monthly_predefined_features(nb_months):
    """
    Get predefined tsfresh features for monthly processing based on nb_months.
    Returns dictionary with configurations for _curr_M, _6_M, and _12_M suffixes.
    """
    features_curr_M_list = get_features_curr_M_list()
    features_6_M_list = get_features_6_M_list() 
    features_12_M_list = get_features_12_M_list()
    
    predefined_features_curr_M = create_tsfresh_config_from_feature_list(features_curr_M_list)
    predefined_features_6_M = create_tsfresh_config_from_feature_list(features_6_M_list)
    predefined_features_12_M = create_tsfresh_config_from_feature_list(features_12_M_list)
    
    return {
        "_curr_M": predefined_features_curr_M,
        "_6_M": predefined_features_6_M,
        "_12_M": predefined_features_12_M
    }