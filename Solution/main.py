########## FG4 - MarkovModelficator with Lasso Attachment ##########


import dataconnection as dc
import dataprep as prep
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from MarkovPredictions import MarkovPredictions

def create_regression_dataframe(data_source):
    regression_df = prep.double_mad_filter(data_source)
    for pred_var in dc.predictive_variables:
        model_set = MarkovPredictions(pred_var=pred_var, data_source=data_source)
        model_set.build_predictions()
        regression_df = regression_df.merge(
            model_set.full_prediction_df,
            left_on=[pred_var, dc.current_state_field], right_on=[pred_var, 'states']
        )
        column_names = list(regression_df.columns)
        column_names[-1] = '%s%s' % (pred_var, 'Prediction')
        regression_df.columns = column_names
    regression_df.drop(
        dc.predictive_variables + [
            'states_x', 'states_y', 'states'
        ],
        axis=1,
        inplace=True
    )
    return regression_df

def create_learning_sets(state, data_source):
    df_state = data_source.loc[
        data_source[dc.current_state_field]==state
    ].copy()
    df_state.drop(
        [dc.current_state_field], axis=1, inplace=True
    )
    X, y = df_state[
        [
            '%sPrediction' % (pred_var) for pred_var in dc.predictive_variables
                ]
            ].values, \
           df_state[dc.state_duration_field].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    return X_train, X_test, y_train, y_test

def main():
    print('\nCreating DataFrame for regression. Following steps are executed:')
    print(' 1. Markov models are generated for each value of predictive variables.')
    print(' 2. DataFrame is constructed by removing outliers and bringing Markov predictions into data as regressors.')
    connection = dc.connect_to_data()

    print('Compiling the Markov data source...')
    markov_data_source = pd.read_sql_query(
        sql=dc.pf_query,
        con=connection
    )
    print('done!')
    df_model = create_regression_dataframe(markov_data_source)
    print('\nRegression DataFrame successfully built.')
    df_r2s = []
    print('\nPerforming cross-validated Lasso regression for each patient flow state.')
    for state in list(df_model[dc.current_state_field].unique()):
        print('Performing cross-validation for %s' % (state))
        X_train, X_test, y_train, y_test = create_learning_sets(state=state, data_source=df_model)
        if X_train.size > 0:
            lasso_reg = LassoCV(
                eps=.001,
                n_alphas=100,
                alphas=None,
                fit_intercept=True,
                normalize=True,
                precompute='auto',
                max_iter=1000,
                tol=.0001,
                copy_X=True,
                cv=5,
                verbose=False,
                n_jobs=-1,
                positive=True,
                random_state=None,
                selection='random'
            )
            lasso_reg.fit(X_train,y_train)
            train_score = lasso_reg.score(X_train, y_train),
            test_score = lasso_reg.score(X_test, y_test),
            y_test_pred = lasso_reg.predict(X_test)
            y_train_pred = lasso_reg.predict(X_train)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
        else:
            train_score = 0
            test_score = 0
            train_mae = 0
            test_mae = 0
        df_r2 = pd.DataFrame()
        df_r2['State'] = str(state)
        df_r2['Rsquared_Train'] = train_score
        df_r2['RSquared_Test'] = test_score
        df_r2['MAE_Train'] = train_mae
        df_r2['MAE_Test'] = test_mae
        df_r2.head()
        df_r2 = df_r2.reset_index()
        df_r2s.append(df_r2)
    df_out = pd.concat(df_r2s, ignore_index=True)
    df_out = df_out.reset_index()
    df_out_test = df_r2s[2]
    df_out.to_csv(path_or_buf='out.csv', sep='|', index_label='index')
    df_out_test.to_csv(path_or_buf='out_test.csv', sep='|', index_label='index')
    return

if __name__ == "__main__":
    main()

#for pred_var in dc.predictive_variables:
#    model_set = MarkovPredictions(pred_var=pred_var, data_source=markov_data_source)