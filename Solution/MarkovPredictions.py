# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:02:17 2017

@author: Tyler Hughes
"""
from makemodel import make_model
import pandas as pd


class MarkovPredictions:
    def __init__(self, pred_var, data_source):
        self.pred_var = pred_var
        self.data_source = data_source
        self.predictions = []
        self.full_prediction_df = pd.DataFrame()

    def build_predictions(self, n_iter=10):
        print('\nBuilding predictions for %s\n' % self.pred_var)
        for value in list(self.data_source[self.pred_var].unique()):
            print('Averaging models for %s... ' % value)
            predictions = []
            for i in range(n_iter):
                model = make_model(data_source=self.data_source, pred_var=self.pred_var, pred_val=value)
                if model is None:
                    break
                if model.singular_matrix is True:
                    print('Singular matrix detected! model for %s cannot be created.' % model.pred_val)
                else:
                    predictions.append(model.prediction_df)
            print('done!')
            if predictions:
                df_averaged = pd.concat(predictions, ignore_index=True).groupby(['states']).mean()
                df_averaged = df_averaged.reset_index()
                df_averaged[self.pred_var] = value
                self.predictions.append(df_averaged)
        self.full_prediction_df = pd.concat(self.predictions)
        return self
