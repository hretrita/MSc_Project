## WEIGHTED CLASS VOTE

import numpy as np
import pandas as pd

# Import files and establish variables
ABCpred = pd.read_csv('abcpred_holdout.csv')
LBtope = pd.read_csv('lbtope_holdout.csv')
iBCE_EL = pd.read_csv('ibceel_holdout.csv')
Bepipred2 = pd.read_csv('bepipred2_holdout.csv')
ground_truth = pd.read_csv('./ground_truth/holdout/02_holdout.csv')
ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
ground_truth = ground_truth.reset_index()
ground_truth = ground_truth.drop(['index'], axis=1)

# ABCpred = pd.read_csv('abcpred_training.csv')
# LBtope = pd.read_csv('lbtope_training.csv')
# iBCE_EL = pd.read_csv('ibceel_training.csv')
# Bepipred2 = pd.read_csv('bepipred2_training.csv')
# ground_truth = pd.read_csv('./ground_truth/training/01_training.csv')
# ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
# ground_truth = ground_truth.reset_index()
# ground_truth = ground_truth.drop(['index'], axis=1)

tables = []
results = []
probs = []
pred = []
threshold = 0.5

# Set the weights in the respective order of the predictors below (i.e., ABCpred, Bepipred2, iBCE-EL, LBtope)
weights = [0.41, 0.55, 0.59, 0.55]
weights_sum = sum(weights)

# Scale weights
ABCpred_weight = weights[0]/weights_sum
Bepipred2_weight = weights[1]/weights_sum
iBCE_EL_weight = weights[2]/weights_sum
LBtope_weight = weights[3]/weights_sum

raw_models = {'ABCpred': [ABCpred, ABCpred_weight], 'Bepipred2': [Bepipred2, Bepipred2_weight],
              'iBCE-EL': [iBCE_EL, iBCE_EL_weight],'LBtope': [LBtope, LBtope_weight]}

# Multiply class values by scaled weights
for table in raw_models.values():
    processed_table = table[0].sort_values(by=['Info_protein_id', 'Info_pos'])
    results.append(processed_table.iloc[:, -1].values * table[1])
    probs.append(processed_table.iloc[:, -2].values)
    final_table = processed_table.iloc[:, 0:3]  # Add cols Info_protein_id and Info_pos

# Concatenate final_table to ground_truth, remove NAs and drop ground_truth column
final_table = final_table.reset_index()
final_table = final_table.drop(['index'], axis=1)
final_table = pd.concat([final_table, ground_truth.iloc[:, -1]], axis=1) #.dropna()
final_table = final_table.dropna()
final_table = final_table.iloc[:, :-1]

# Transpose matrix
results = np.transpose(np.stack(tuple(results)))
probs = np.transpose(np.stack(tuple(probs)))

# Weighted Vote Implementation
for idx, r in enumerate(results):
    sum = r.sum() # Sum of the row
    if sum > 0:
        pred.append(1)
    else:
        pred.append(-1)

# Merge predictions with true class and remove NaN
pred = pd.DataFrame(pred)
pred = pred.reset_index()
pred = pred.drop(['index'], axis=1)
ground_truth_and_pred = pd.concat([ground_truth, pred], axis=1)
ground_truth_and_pred = ground_truth_and_pred.dropna()

# Split true class and predictions
pred = ground_truth_and_pred.iloc[:, -1]
ground_truth = ground_truth_and_pred.iloc[:, -2]

# Concatenate predictions to final_table and export csv file ready to run gather_results in R
final_table['pred'] = pred
final_table.to_csv('./ensemble_preds/04_Spyogenes/holdout/w_mv.csv', index=False)
