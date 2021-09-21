## SIMPLE AVERAGE PREDICTED PROBABILITY

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from os import listdir

# Establish variables
files = listdir("./") # .csv files must be in the set directory
# true class data is stored in a separate folder
ground_truth = pd.read_csv('./ground_truth/training/01_training.csv')
# Reorder GT in alphabetical order for consistency with other dataframes
ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
ground_truth = ground_truth.reset_index()
ground_truth = ground_truth.drop(['index'], axis=1)

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
    # Add cols Info_protein_id and Info_pos
    final_table = tables[table_id].iloc[:, 0:3]

# Concatenate final_table and ground_truth, remove NAs and drop ground_truth column
final_table = final_table.reset_index()
final_table = final_table.drop(['index'], axis=1)
final_table = pd.concat([final_table, ground_truth.iloc[:, -1]], axis=1) #.dropna()
final_table = final_table.dropna()
final_table = final_table.iloc[:, :-1]

# Transpose matrix
results = np.transpose(np.stack(tuple(predictions)))
probs = np.transpose(np.stack(tuple(probs)))

# Average Predicted Probability Implementation
divisor = probs.shape[1] # Number of predictors being used. Used to compute average
for idx, p in enumerate(probs):
    sum = p.sum()
    if sum/divisor < threshold:
        pred.append(-1)
    else:
        pred.append(1)

#Merge predictions with ground_truth and remove NA
pred = pd.DataFrame(pred)
pred = pred.reset_index()
pred = pred.drop(['index'], axis=1)
ground_truth_and_pred = pd.concat([ground_truth, pred], axis=1)
ground_truth_and_pred = ground_truth_and_pred.dropna()

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

# Concatenate predictions to final_table and export csv file ready to run gather_results in R
final_table['pred'] = pred
final_table.to_csv('./ensemble_preds/04_Spyogenes/training/s_app.csv', index=False)
