import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

# Import necessary files
# ABCpred = pd.read_csv('abcpred_holdout.csv')
# LBtope = pd.read_csv('lbtope_holdout.csv')
# iBCE_EL = pd.read_csv('ibceel_holdout.csv')
# Bepipred2 = pd.read_csv('bepipred2_holdout.csv')
# ground_truth = pd.read_csv('./ground_truth/holdout/02_holdout.csv')
# ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
# ground_truth = ground_truth.reset_index()
# ground_truth = ground_truth.drop(['index'], axis=1)
# pep_gt = pd.read_csv('./res_files/01_EBV/base/01_EBV_res_h_abcpred.csv')
# pep_gt = pep_gt.filter(['Class'], axis=1) # ground truth at peptide level

ABCpred = pd.read_csv('abcpred_training.csv')
LBtope = pd.read_csv('lbtope_training.csv')
iBCE_EL = pd.read_csv('ibceel_training.csv')
Bepipred2 = pd.read_csv('bepipred2_training.csv')
ground_truth = pd.read_csv('./ground_truth/training/01_training.csv')
ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
ground_truth = ground_truth.reset_index()
ground_truth = ground_truth.drop(['index'], axis=1)
pep_lvl = pd.read_csv('./res_files/01_EBV/base/01_EBV_res_t_abcpred.csv')
pep_gt = pep_lvl.filter(['Class'], axis=1) # ground truth at peptide level
pep_pred = pep_lvl.filter(['Pred'], axis=1) # prediction at peptide level

tables = [ABCpred, LBtope, iBCE_EL, Bepipred2]
probs = []

# Order files by col1 and col2
for table_id in range(len(tables)):
    tables[table_id] = tables[table_id].sort_values(by=['Info_protein_id', 'Info_pos'])
    # Save predictions in predictions list and probabilities in probs list
    probs.append(tables[table_id].iloc[:, -2].values)

# Transpose matrix
probs = np.transpose(np.stack(tuple(probs)))
probs = pd.DataFrame(probs)

# Order files by col1 and col2 and remove entries with no ground truth
frames = [probs, ground_truth]
data = pd.concat([probs, ground_truth], axis=1)
data = data.dropna()
df = data

##################### Calculate average probability at peptide level #####################
df = (df
      # Sort the values so everything is in order of protein and position
      .sort_values(['Info_protein_id', 'Info_pos'], ascending=True)

      # reset the index (just to keep things clean)
      .reset_index(drop=True)
      )

# Populate a new column called "section"
df['Info_PepID'] = None

# Populate the necessary variables
last_protein = 'no_protein'
last_position = -1
prev_gt = 'no_class'
Info_PepID = 1

# Iterate over each row using iterrows
for row in df.iterrows():
    protein = row[1]['Info_protein_id']
    position = row[1]['Info_pos']
    gt = row[1]['Class']

    # If the current Class matches the different class
    if gt == prev_gt or prev_gt == 'no_class':

        # If the current protein and last protein match, update the section label of the dataframe
        if protein == last_protein or last_protein == 'no_protein':

            # Check if the current position is a continuation of the last position, if so insert the section label as-is
            if position == last_position + 1 or last_position == -1:
                df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'

            # If not, increase the section variable, and then insert the section label
            else:
                Info_PepID += 2
                df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'

        # If the current protein is a new protein, restart the section naming
        else:
            Info_PepID = 1
            df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'

    # If class changes in the next AA
    else:

        # If the current protein and last protein match, update the section label of the dataframe
        if protein == last_protein or last_protein == 'no_protein':

            # Check if the current position is a continuation of the last position, if so insert the section label as-is
            if position == last_position + 1 or last_position == -1:
                Info_PepID += 1
                df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'

            # If not, increase the section variable, and then insert the section label
            else:
                Info_PepID += 2
                df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'
        else:
            Info_PepID = 1
            df.loc[row[0], 'Info_PepID'] = f'{protein}:{Info_PepID}'

    # After evaluating the current protein, reassign the "last_" variables
    last_protein = protein
    last_position = position
    prev_gt = gt

# The groupby you want to do
output = (
    df[['Info_PepID', 0, 1, 2, 3]]
        .groupby('Info_PepID')
        .mean()
        .reset_index()
)

# Calculate average probability per peptide
output = (df.groupby('Info_PepID')[[0, 1, 2, 3]].mean()) # compute av prob
output = output.reset_index() # Sets an index and pushes Info_PepID into col1

# Vector of ground truths
y = pep_gt
# Vector of probabilities
prob = output.iloc[: , 1:]

# fpr, tpr, thresholds = metrics.roc_curve(y, prob.filter([0]), pos_label=1)
fpr, tpr, thresholds = metrics.roc_curve(y, pep_pred, pos_label=1)

auc = metrics.auc(fpr, tpr)

# auc = roc_auc_score(y, prob)
print(auc)

# auc1 = roc_auc_score(y, prob)
#
# print(auc1)
# print(auc)