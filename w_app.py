import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

#Import files
ABCpred = pd.read_csv('abcpred_holdout.csv')
LBtope = pd.read_csv('lbtope_holdout.csv')
iBCE_EL = pd.read_csv('ibceel_holdout.csv')
Bepipred2 = pd.read_csv('bepipred2_holdout.csv')
ground_truth = pd.read_csv('./ground_truth/holdout/02_holdout.csv')
ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])

# ABCpred = pd.read_csv('abcpred_training.csv')
# LBtope = pd.read_csv('lbtope_training.csv')
# iBCE_EL = pd.read_csv('ibceel_training.csv')
# Bepipred2 = pd.read_csv('bepipred2_training.csv')
# ground_truth = pd.read_csv('./ground_truth/training/01_training.csv')
# ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])

# Make a list of files
tables = []
results = []
probs = []
pred = []
threshold = 0.5

# Set the weights in the respective order of the predictors below (i.e., ABCpred, LBtope, iBCE-EL, Bepipred2)
weights = [0.50,	0.50,	0.50,	0.80]
weights_sum = sum(weights)

ABCpred_weight = weights[0]/weights_sum
Bepipred2_weight = weights[1]/weights_sum
iBCE_EL_weight = weights[2]/weights_sum
LBtope_weight = weights[3]/weights_sum

raw_models = {'ABCpred': [ABCpred, ABCpred_weight], 'Bepipred2': [Bepipred2, Bepipred2_weight],
              'iBCE-EL': [iBCE_EL, iBCE_EL_weight],'LBtope': [LBtope, LBtope_weight]}

# Weighted Average Predicted Probability Implementation
# Multiply class values by scaled weights
for table in raw_models.values():
    processed_table = table[0].sort_values(by=['Info_protein_id', 'Info_pos'])
    results.append(processed_table.iloc[:, -1].values)
    probs.append(processed_table.iloc[:, -2].values * table[1])
    final_table = processed_table.iloc[:, 0:2]  # Add cols Info_UID and Info_center_pos

# Concatenate final_table to ground_truth, remove NAs and drop ground_truth column
final_table = pd.concat([final_table, ground_truth.iloc[:,-1]], axis=1).dropna()
final_table = final_table.iloc[:, :-1]
final_table = final_table.sort_values(by=['Info_protein_id', 'Info_pos'])


# Transpose matrix
results = np.transpose(np.stack(tuple(results)))
probs = np.transpose(np.stack(tuple(probs)))

# Classification
#divisor = probs.shape[1]
for idx, p in enumerate(probs):
    sum = p.sum()
    if sum < threshold:
        pred.append(-1)
    else:
        pred.append(1)

# Merge predictions with true class and remove NaN
pred = pd.DataFrame(pred)
ground_truth_and_pred = pd.concat([ground_truth, pred], axis=1).dropna()
# Split true class and predictions
pred = ground_truth_and_pred.iloc[:, -1]
ground_truth = ground_truth_and_pred.iloc[:, -2]

# Test set performance
accuracy = accuracy_score(ground_truth, pred) # Accuracy
mcc = matthews_corrcoef(ground_truth, pred) # MCC
f1 = f1_score(ground_truth, pred) # F1-score

print('Model performance for test set')
print('- Accuracy: %s' % accuracy)
print('- MCC: %s' % mcc)
print('- F1 score: %s' % f1)

# Create final table with results
final_table['pred'] = pred
final_table.to_csv('./ensemble_preds/04_Spyogenes/holdout/w_app.csv', index=False)

