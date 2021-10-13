# Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import files and establish variables
# ABCpred = pd.read_csv('abcpred_holdout.csv')
# LBtope = pd.read_csv('lbtope_holdout.csv')
# iBCE_EL = pd.read_csv('ibceel_holdout.csv')
# Bepipred2 = pd.read_csv('bepipred2_holdout.csv')
# ground_truth = pd.read_csv('./ground_truth/holdout/02_holdout.csv')
# ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
# ground_truth = ground_truth.reset_index()
# ground_truth = ground_truth.drop(['index'], axis=1)

ABCpred = pd.read_csv('abcpred_training.csv')
LBtope = pd.read_csv('lbtope_training.csv')
iBCE_EL = pd.read_csv('ibceel_training.csv')
Bepipred2 = pd.read_csv('bepipred2_training.csv')
ground_truth = pd.read_csv('./ground_truth/training/01_training.csv')
ground_truth = ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
ground_truth = ground_truth.reset_index()
ground_truth = ground_truth.drop(['index'], axis=1)

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
# Remove unnecessary columns. Only keep probabilities and ground truth
data = data.drop(['Info_protein_id', 'Info_pos', 'Info_AA'], axis=1)

# Separate features from label
X, y = data.iloc[:, :-1], data.iloc[:, [-1]]

#===================== ML ALGORITHMS =============================
# Random Forest
rf = RandomForestClassifier(class_weight='balanced',random_state=42)
rf.fit(X, y.values.ravel())

#=================================================================
# Support Vector Machine
svm = SVC(class_weight='balanced')
svm.fit(X, y.values.ravel())

#=================================================================
# Logistic Regression
lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X, y.values.ravel())

############### Validation ###############

# Establish variables
v_ABCpred = pd.read_csv('abcpred_holdout.csv')
v_LBtope = pd.read_csv('lbtope_holdout.csv')
v_iBCE_EL = pd.read_csv('ibceel_holdout.csv')
v_Bepipred2 = pd.read_csv('bepipred2_holdout.csv')
v_ground_truth = pd.read_csv('./ground_truth/holdout/02_holdout.csv')
v_ground_truth = v_ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
v_ground_truth = v_ground_truth.reset_index()
v_ground_truth = v_ground_truth.drop(['index'], axis=1)

# v_ABCpred = pd.read_csv('abcpred_training.csv')
# v_LBtope = pd.read_csv('lbtope_training.csv')
# v_iBCE_EL = pd.read_csv('ibceel_training.csv')
# v_Bepipred2 = pd.read_csv('bepipred2_training.csv')
# v_ground_truth = pd.read_csv('./ground_truth/training/01_training.csv')
# v_ground_truth = v_ground_truth.sort_values(by=['Info_protein_id', 'Info_pos'])
# v_ground_truth = v_ground_truth.reset_index()
# v_ground_truth = v_ground_truth.drop(['index'], axis=1)

v_tables = [v_ABCpred, v_LBtope, v_iBCE_EL, v_Bepipred2]
v_probs = []

# Order files by col1 and col2
for table_id in range(len(v_tables)):
    v_tables[table_id] = v_tables[table_id].sort_values(by=['Info_protein_id', 'Info_pos'])
    # Save predictions in predictions list and probabilities in probs list
    v_probs.append(v_tables[table_id].iloc[:, -2].values)

# Create final table where predictions will be added to
temp = v_ground_truth.dropna().drop(['Class'], axis=1)

# Drop unnecessary cols - keep ground truth classifications
v_ground_truth_tmp = v_ground_truth.drop(['Info_pos', 'Info_AA'], axis=1)
v_ground_truth = v_ground_truth.drop(['Info_protein_id', 'Info_pos', 'Info_AA'], axis=1)
v_ground_truth = v_ground_truth.dropna()

# Transpose matrix
v_probs = np.transpose(np.stack(tuple(v_probs)))
v_probs = pd.DataFrame(v_probs)

# Make prediction with chosen model
prediction = lr.predict(v_probs) # <---- Change classification model if needed
prediction = np.transpose(np.stack(tuple(prediction)))
prediction = pd.DataFrame(prediction)
pred = pd.concat([prediction, v_ground_truth], axis=1).dropna()

# Concatenate predictions to temp and export csv file ready to run gather_results in R
temp['pred'] = pred.iloc[:,0]
temp.to_csv('./stacking/04_Spyogenes/hot_lr_stacking.csv', index=False)
