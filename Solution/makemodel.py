# -*- coding: utf-8 -*-

from dataconnection import \
    current_state_field, \
    next_state_field, \
    state_duration_field, \
    absorbing_state
from MarkovModel import MarkovModel
import dataprep as prep


def make_model(data_source, pred_var, pred_val):
    model = MarkovModel(
                        sample=prep.bootstrap_sampler(
                            data_source=prep.filter_data(data_source, pred_var),
                            pred_var=pred_var,
                            pred_val=pred_val,
                            replace=True,
                            frac=0.3
                        ),
                        pred_var=pred_var,
                        pred_val=pred_val,
                        absorbing_state=absorbing_state,
                        current_state_field=current_state_field,
                        next_state_field=next_state_field,
                        state_duration_field=state_duration_field
    )
    model.calculate_average_state_duration()
    model.calculate_transition_probabilities()
    if absorbing_state not in list(model.transition_probabilities[model.current_state_].unique()):
        print('%s never entered absorbing state. Model cannot be created.' % pred_val)
        return None
    else:
        model.make_P_matrix()
        model.check_data_validity()
        model.find_fundamental_matrix()
        if model.singular_matrix:
            print('The Q-matrix is singular and cannot be inverted. Model cannot be created.')
            return None
        else:
            model.consolidate_model_parameters()
            return model
