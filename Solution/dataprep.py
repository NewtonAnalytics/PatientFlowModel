from dataconnection import \
    current_state_field, \
    state_duration_field, \
    key_field, \
    predictive_variables
import pandas as pd
import numpy as np

filter_by = 'LocationCode'


def double_mad_filter(data_source, pred_var=filter_by):
    df_medians = data_source.copy()
    df_medians = df_medians.groupby([current_state_field, pred_var]).median()
    df_medians = df_medians.reset_index()
    df_medians.columns = [current_state_field, pred_var, 'MedianDuration']

    df_low = data_source.copy()
    df_low = df_low.merge(df_medians, on=[current_state_field, pred_var])
    df_low = df_low[df_low[state_duration_field] <= df_low.MedianDuration]
    df_mad_low = df_low.copy()
    df_mad_low = df_mad_low.groupby([current_state_field, pred_var]).mad()
    df_mad_low = df_mad_low.reset_index()
    df_mad_low.drop(['MedianDuration'], axis=1, inplace=True)
    df_mad_low.columns = [current_state_field, pred_var, 'MADLowDuration']

    df_high = data_source.copy()
    df_high = df_high.merge(df_medians, on=[current_state_field, pred_var])
    df_high = df_high[df_high[state_duration_field] > df_high.MedianDuration]
    df_mad_high = df_high.copy()
    df_mad_high = df_mad_high.groupby([current_state_field, pred_var]).mad()
    df_mad_high = df_mad_high.reset_index()
    df_mad_high.drop(['MedianDuration'], axis=1, inplace=True)
    df_mad_high.columns = [current_state_field, pred_var, 'MADHighDuration']

    df_high = df_high.merge(df_mad_high, on=[current_state_field, pred_var])
    df_high = df_high.reset_index()
    df_low = df_low.merge(df_mad_low, on=[current_state_field, pred_var])
    df_low = df_low.reset_index()

    consistent_names = list(df_high.columns)
    consistent_names.remove('MADHighDuration')
    consistent_names.append('MADDuration')
    df_high.columns = consistent_names
    df_low.columns = consistent_names
    df_master = pd.concat([df_high, df_low])

    df_master['NumberOfDeviationsFromMedian'] = abs(
        df_master[state_duration_field] - df_master.MedianDuration
    ) / df_master.MADDuration
    df_master.fillna(value=0, inplace=True)
    df_inliers = df_master.loc[
        (df_master.NumberOfDeviationsFromMedian <= 2) | (df_master.NumberOfDeviationsFromMedian == np.inf)
    ]
    df_inliers = df_inliers.groupby(predictive_variables + [key_field, current_state_field]).sum()
    df_inliers = df_inliers.reset_index()
    nonpredictive_variables = list(df_inliers.columns)
    regression_variables = predictive_variables + [state_duration_field, current_state_field]
    df_inliers.drop([x for x in nonpredictive_variables if x not in regression_variables], axis=1, inplace=True)
    return df_inliers


def bootstrap_sampler(data_source, pred_var, pred_val, replace=True, frac=0.3):
    df_sample = data_source[data_source[pred_var] == pred_val].filter(items=['pf_enc', pred_var]).reset_index()
    if 'index' in list(df_sample.columns):
        df_sample = df_sample.drop('index', axis=1)
    df_sample = df_sample.drop_duplicates().reset_index().sample(frac=frac, replace=replace)
    df_source = data_source.copy()
    df_sampled = df_source.merge(df_sample, left_on=['pf_enc', pred_var], right_on=['pf_enc', pred_var], how='inner')
    df_sampled = df_sampled.drop(['index_x', 'index_y'], axis=1)
    df_sample = df_sampled.copy()
    return df_sample


def filter_data(data_source, pred_var):
    nonpred_vars = [x for x in list(data_source.columns) if x not in predictive_variables]
    relevant_columns = list(nonpred_vars) + [pred_var]
    filtered_df = data_source.filter(items=relevant_columns, axis=1).reset_index().copy()
    return filtered_df