## SIMPLE MAJORITY VOTE

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from os import listdir

# Establish variables
files = listdir("./") # .csv files must be in the set directory
# true class data is stored in a separate folder
ground_truth = pd.read_csv('./ground_truth/holdout/02_holdout.csv')
# Reorder GT in alphabetical order for consistency with other dataframes
ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
ground_truth = ground_truth.reset_index()

tables = []
predictions = []
probs = []
pred = []
threshold = 0.5

# Save necessary files in tables list. The only .csv files in the folder should be the results of the 4 predictors
for filename in files:
    if "." in filename:
        if filename.split('.')[1] == 'csv':
            tables.append(pd.read_csv(filename))

# Order files by col1 and col2
for table_id in range(len(tables)):
    # Order df in alphabetical order
    tables[table_id] = tables[table_id].sort_values(by=['Info_protein_id', 'Info_pos'])
    # Save predictions in predictions list, probabilities in probs list, temporary final table
    predictions.append(tables[table_id].iloc[:, -1].values)
    probs.append(tables[table_id].iloc[:, -2].values)
    final_table = tables[table_id].iloc[:, 0:2]  # Add cols Info_protein_id and Info_pos

# Concatenate final_table and ground_truth, remove NAs and drop ground_truth column
final_table = pd.concat([final_table, ground_truth.iloc[:, -1]], axis=1) #.dropna()
final_table = final_table.dropna()
final_table = final_table.iloc[:, :-1]

# Transpose matrix
results = np.transpose(np.stack(tuple(predictions)))
probs = np.transpose(np.stack(tuple(probs)))

# Majority Voting Implementation
for idx, r in enumerate(results):
    sum = r.sum()
    if sum < 0:
        pred.append(-1)
    elif sum == 0:
        avg_prob = probs[idx].sum()/len(probs[idx]) # In case the sum is 0, the average of the probability will be computed and classified with respect to the threshold
        if avg_prob >= threshold:
            pred.append(1)
        else:
            pred.append(-1)
    else:
        pred.append(1)

# Merge predictions with ground_truth and remove NA
pred = pd.DataFrame(pred)
ground_truth_and_pred = pd.concat([ground_truth, pred], axis=1).dropna()
# Split true class and predictions
pred = ground_truth_and_pred.iloc[:, -1]
ground_truth_ = ground_truth_and_pred.iloc[:, -2]

# Majority Vote Performance Metrics
accuracy = accuracy_score(ground_truth_, pred) # Accuracy
mcc = matthews_corrcoef(ground_truth_, pred) # MCC
f1 = f1_score(ground_truth_, pred) # F1-score

print('Model performance for test set')
print('- Accuracy: %s' % accuracy)
print('- MCC: %s' % mcc)
print('- F1 score: %s' % f1)

# Concatenate predictions to final_table and export csv file ready to run gather_results in R
final_table['pred'] = pred
final_table.to_csv('./ensemble_preds/02_HepC/holdout/s_mv.csv', index=False)