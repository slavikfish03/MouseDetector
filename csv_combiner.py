import pandas as pd
import numpy as np
import os
from pykalman import KalmanFilter


BLOCK_SIZE = 25
OBSERVATION_MATRIX = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.5, 0.0, 0.5],
])


class CSVCombiner:
    def __init__(self, path_to_static_data, path_to_behavior_data, shift):
        self.path_to_static_data = path_to_static_data
        self.path_to_behavior_data = path_to_behavior_data
        self.shift = shift

    def combine(self):
        df_static = pd.read_csv(f'{self.path_to_static_data}.csv')
        df_beh = pd.read_csv(f'{self.path_to_behavior_data}.csv')
        df_beh = self.calman(df_beh)
        df_beh = self.smooth(df_beh)

        df_beh_shifted = pd.DataFrame(np.nan, index=range(self.shift), columns=df_beh.columns)
        df_beh_shifted = pd.concat([df_beh_shifted, df_beh], ignore_index=False)
        df_beh_shifted.index = range(len(df_beh_shifted))

        result = pd.concat([df_static, df_beh_shifted], axis=1)

        os.makedirs('mouse_data', exist_ok=True)
        excel_file = f'mouse_data\\{self.path_to_behavior_data[:len(self.path_to_behavior_data) - 4]}_data.xlsx'
        result.to_excel(excel_file, index=False)

    def calman(self, df_beh):
        num_features = df_beh.shape[1]
        initial_state = df_beh.iloc[0].values

        transition_matrix = np.eye(num_features)
        Q = np.eye(num_features) * 0.01
        R = np.eye(num_features) * 0.1

        kf = KalmanFilter(transition_matrices=transition_matrix,
                          observation_matrices=OBSERVATION_MATRIX,
                          initial_state_mean=initial_state,
                          transition_covariance=Q,
                          observation_covariance=R)
        filtered_state_means, _ = kf.filter(df_beh.values)
        filtered_df = pd.DataFrame(filtered_state_means, columns=df_beh.columns)
        filtered_df = np.clip(filtered_df, 0, 1)
        filtered_df = filtered_df.div(filtered_df.sum(axis=1), axis=0)
        return filtered_df

    def smooth(self, df_beh):
        df_beh_boolean = pd.concat([self.smooth_for_block(df_beh.iloc[i : i + BLOCK_SIZE]) for i in range(0, len(df_beh), BLOCK_SIZE)], axis=0)
        return df_beh_boolean

    def smooth_for_block(self, block):
        max_indices = block.idxmax(axis=1)
        max_count = max_indices.value_counts()

        if len(max_count[max_count == max_count.max()]) > 1:
            behs = max_count[max_count == max_count.max()].index
            avg_probs = block[behs].mean()
            selected_beh = avg_probs.idxmax()
        else:
            selected_beh = max_count.idxmax()

        block_boolean = pd.DataFrame(False, index=block.index, columns=block.columns)
        block_boolean[selected_beh] = True
        return block_boolean