# import pandas as pd
# import numpy as np
#
# # d = {'Info_protein_id': ['Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 1', 'Protein 2', 'Protein 2', 'Protein 2', 'Protein 2', 'Protein 2', 'Protein 2', 'Protein 3', 'Protein 3'],
# #      'Info_pos_AA': [22, 23, 24, 25, 26, 34, 35, 36, 37, 38, 45, 46, 47, 48, 49, 55, 56, 57],
# #      0: [0.4, 0.3, 0.7, 0.6, 0.4, 0.2, 0.6, 0.7, 0.8, 0.7, 0.7, 0.2, 0.6, 0.7, 0.8, 0.7, 0.7, 0.2],
# #      1: [0.5, 0.7, 0.8, 0.9, 0.9, 0.1, 0.6, 0.5, 0.7, 0.5, 0.4, 0.4, 0.8, 0.9, 0.9, 0.1, 0.6, 0.5],
# #      'Class': [1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1]
# #      }
# df = pd.read_csv('bepipred2_training.csv')
# gt = pd.read_csv('./ground_truth/training/01_training.csv')
# gt = gt.sort_values(by=['Info_protein_id', 'Info_pos'])
# gt = gt.reset_index()
# gt = gt.drop(['index'], axis=1)
#
# data = pd.concat([df, gt.iloc[:,-1]], axis=1)
# data = data.dropna()
# data = data.drop(['pred'], axis = 1)
#
# df = data
#
# ##################### Calculate average probability at peptide level #####################
# df = (df
#       # Sort the values so everything is in order of protein and position
#       .sort_values(['Info_protein_id', 'Info_pos'], ascending=True)
#
#       # reset the index (just to keep things clean)
#       .reset_index(drop=True)
#       )
#
# # Populate a new column called "section"
# df['Info_PepID'] = None
#
# # Populate the necessary variables
# last_protein = 'no_protein'
# last_position = -1
# prev_gt = 'no_class'
# Info_PepID = 1
#
# # Iterate over each row using iterrows
# for row in df.iterrows():
#     protein = row[1]['Info_protein_id']
#     position = row[1]['Info_pos']
#     gt = row[1]['Class']
#
#     # If the current Class matches the different class
#     if gt == prev_gt or prev_gt == 'no_class':
#
#         # If the current protein and last protein match, update the section label of the dataframe
#         if protein == last_protein or last_protein == 'no_protein':
#
#             # Check if the current position is a continuation of the last position, if so insert the section label as-is
#             if position == last_position + 1 or last_position == -1:
#                 df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'
#
#             # If not, increase the section variable, and then insert the section label
#             else:
#                 Info_PepID += 2
#                 df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'
#
#         # If the current protein is a new protein, restart the section naming
#         else:
#             Info_PepID = 1
#             df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'
#
#     # If class changes in the next AA
#     else:
#
#         # If the current protein and last protein match, update the section label of the dataframe
#         if protein == last_protein or last_protein == 'no_protein':
#
#             # Check if the current position is a continuation of the last position, if so insert the section label as-is
#             if position == last_position + 1 or last_position == -1:
#                 Info_PepID += 1
#                 df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'
#
#             # If not, increase the section variable, and then insert the section label
#             else:
#                 Info_PepID += 2
#                 df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'
#         else:
#             Info_PepID = 1
#             df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'
#
#     # After evaluating the current protein, reassign the "last_" variables
#     last_protein = protein
#     last_position = position
#     prev_gt = gt
#
# # The groupby you want to do
# desired_output = (
#     df[['Info_PepID', 'prob']]
#         .groupby('Info_PepID')
#         .mean()
#         .reset_index()
# )
#
# print(df)
# av = (df.groupby('Info_PepID')[['prob']].mean()) # compute av prob
# print(av.to_string())
# print(av)