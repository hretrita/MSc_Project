import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from os import listdir

# Establish variables
files = listdir("./") # .csv files must be in the set directory
ground_truth = pd.read_csv('./ground_truth/holdout/02_holdout.csv') # true class data is stored in a separate folder
ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
tables = []
predictions = []
probs = []
pred = []
threshold = 0.5

# Save necessary files in tables list. Only parsed data should be in the directory as a .csv file
for filename in files:
    if "." in filename:
        if filename.split('.')[1] == 'csv':
            tables.append(pd.read_csv(filename))

# Order files by col1 and col2
for table_id in range(len(tables)):
    tables[table_id] = tables[table_id].sort_values(by=['Info_protein_id', 'Info_pos'])
    # Save predictions in predictions list and probabilities in probs list
    predictions.append(tables[table_id].iloc[:, -1].values)
    probs.append(tables[table_id].iloc[:, -2].values)
    final_table = tables[table_id].iloc[:, 0:2]  # Add cols Info_UID and Info_center_pos

# Concatenate final_table to ground_truth, remove NAs and drop ground_truth column
final_table = pd.concat([final_table, ground_truth.iloc[:,-1]], axis=1).dropna()
final_table = final_table.iloc[:, :-1]
final_table = final_table.sort_values(by=['Info_protein_id', 'Info_pos'])

# Transpose matrix
results = np.transpose(np.stack(tuple(predictions)))
probs = np.transpose(np.stack(tuple(probs)))

# # Scale every row of the probs table
# probs_tmp = pd.DataFrame(probs)
# probs_tmp.loc[:, :] = probs_tmp.loc[:, :].div(probs_tmp.sum(axis=1), axis=0)
# probs = probs_tmp.to_numpy()

# Average Predicted Probability Implementation
divisor = probs.shape[1] # Number of predictors being used. Used to compute average
for idx, p in enumerate(probs):
    sum = p.sum()
    if sum/divisor < threshold:
        pred.append(-1)
    else:
        pred.append(1)

# Merge predictions with true class and remove NaN
pred = pd.DataFrame(pred)
ground_truth_and_pred = pd.concat([ground_truth, pred], axis=1).dropna()
# Split true class and predictions
pred = ground_truth_and_pred.iloc[:, -1]
ground_truth_ = ground_truth_and_pred.iloc[:, -2]

# Average Predicted Probability Metrics
accuracy = accuracy_score(ground_truth_, pred) # Accuracy
mcc = matthews_corrcoef(ground_truth_, pred) # MCC
f1 = f1_score(ground_truth_, pred) # F1-score

print('Model performance for test set')
print('- Accuracy: %s' % accuracy)
print('- MCC: %s' % mcc)
print('- F1 score: %s' % f1)

# Create final table with results
final_table['pred'] = pred
final_table.to_csv('./ensemble_preds/02_HepC/holdout/s_app.csv', index=False)
