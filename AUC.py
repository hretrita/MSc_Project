import numpy as np
from sklearn import metrics

y = np.array([-1, -1, 1, 1])
prob = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, prob, pos_label=1)
auc = metrics.auc(fpr, tpr)