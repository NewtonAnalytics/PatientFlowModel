import pandas as pd
import numpy as np


class MarkovModel:
    def __init__(self, pred_var, pred_val,
                 absorbing_state, current_state_field,
                 next_state_field, state_duration_field,
                 sample):

        self.pred_var = pred_var
        self.pred_val = pred_val
        self.sample = sample
        self.absorbing_state_ = absorbing_state
        self.current_state_ = current_state_field
        self.next_state_ = next_state_field
        self.state_duration_ = state_duration_field
        self.state_durations = pd.DataFrame()
        self.transition_probabilities = pd.DataFrame()
        self.non_absorbing_states = []
        self.P_matrix = np.zeros(5)
        self.Q_matrix = np.zeros(5)
        self.validity = ''
        self.fundamental_df = pd.DataFrame()
        self.fundamental_matrix = np.zeros(5)
        self.prediction = 0
        self.prediction_df = pd.DataFrame()

    def calculate_average_state_duration(self):
        df_times = self.sample.filter(items=[self.pred_var, self.current_state_, self.state_duration_])
        df_times_grouped = df_times.groupby([self.pred_var, self.current_state_]).mean().reset_index()
        self.state_durations = df_times_grouped.copy()
        return self
    
    def calculate_transition_probabilities(self):
        df_transitions = self.sample.filter(items=[self.pred_var, self.current_state_, self.next_state_])
        df_transitions = df_transitions.groupby([self.pred_var, self.current_state_, self.next_state_]).size().reset_index()
        new_names = list(df_transitions.columns) 
        new_names[-1] = 'transition_count'
        df_transitions.columns = new_names
        df_instances = self.sample.filter(items=[self.pred_var, self.current_state_])
        df_instances = df_instances.groupby([self.pred_var, self.current_state_]).size().reset_index()
        new_names = list(df_instances.columns)
        new_names[-1] = 'instance_count'
        df_instances.columns = new_names
        df_probabilities = df_transitions.merge(df_instances,
                                                left_on=[self.current_state_, self.pred_var],
                                                right_on=[self.current_state_, self.pred_var])
        df_probabilities['transition_probability'] = df_probabilities.transition_count/df_probabilities.instance_count
        df_probabilities_filtered = df_probabilities.filter(items=[self.pred_var, self.current_state_, self.next_state_, 'transition_probability']).reset_index()
        self.transition_probabilities = df_probabilities_filtered.copy()
        return self
    
    def make_P_matrix(self):
        all_states = list(self.transition_probabilities[self.current_state_].unique())
        all_states.remove(self.absorbing_state_)
        self.non_absorbing_states = all_states
        states = self.non_absorbing_states + [self.absorbing_state_]
        self.P_matrix = np.zeros((len(states), len(states)))
        for i in range(len(states)):
            for j in range(len(states)):
                try: 
                    ind_trans_prob = self.transition_probabilities[(self.transition_probabilities[self.current_state_] == states[i]) & (self.transition_probabilities[self.next_state_] == states[j])].iloc[0]['transition_probability']
                    self.P_matrix[i][j] = ind_trans_prob
                except IndexError:
                    self.P_matrix[i][j] = 0
        return self
    
    def check_data_validity(self):
        summed = np.sum(self.P_matrix, axis=1)
        if any(summed < .99) or any(summed > 1.01):
            self.validity='Invalid'
        else:
            self.validity='Valid'
        return self
    
    def find_fundamental_matrix(self):
        self.Q_matrix = self.P_matrix[:-1,:-1]
        try:
            self.fundamental_matrix = np.linalg.inv(np.identity(len(self.non_absorbing_states)) - self.Q_matrix)
            self.singular_matrix=False
            tuples = [(i, j) for i in self.non_absorbing_states for j in self.non_absorbing_states]
            indices = pd.MultiIndex.from_tuples(tuples, names=['starting_from', 'states'])
            fundamental_df = pd.DataFrame(
                self.fundamental_matrix.reshape(len(self.non_absorbing_states)**2,1), index=indices
            )
            fundamental_df = fundamental_df.reset_index()
            fundamental_df.columns = ['starting_from', 'states', 'num_instances']
            self.fundamental_df = fundamental_df.copy()
            self.singular_matrix=False
        except np.linalg.LinAlgError:
            self.singular_matrix=True
        except TypeError:
            self.singular_matrix=True
        return self
    
    def consolidate_model_parameters(self):
        if not self.singular_matrix:
            df_final_model = self.fundamental_df[
                self.fundamental_df['starting_from'] == self.non_absorbing_states[0]
            ].merge(
                self.state_durations,
                left_on=['states'],
                right_on=[self.current_state_]
            )
            df_final_model['predicted_duration'] = df_final_model['num_instances']*df_final_model[self.state_duration_]
            df_final_model = df_final_model.drop(
                [self.pred_var, self.current_state_, 'num_instances', self.state_duration_], axis=1
            )
            self.prediction_df = df_final_model.copy()
            self.prediction = df_final_model.groupby(['starting_from']).sum().iloc[0]['predicted_duration']
        return self
