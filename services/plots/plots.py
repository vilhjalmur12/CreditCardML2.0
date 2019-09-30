import os
import matplotlib.pyplot as plt  # plotting
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, roc_auc_score
import numpy as np
import seaborn as sns
import pandas as pd



def confusion_matrix_plot(model, y_true, y_pred, figsize=(10,5)):
    plt.clf()
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    #fig, ax = plt.subplots(figsize=figsize)
    plt.title = '{} Confusion matrix'.format(model)
    plot = sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='')
    plot.set_title('{} Confusion matrix'.format(model))
    fig = plot.get_figure()
    if not os.path.isdir('./data/results'):
        os.mkdir('./data/results')
    fig.savefig('./data/results/{}_cnf_mtrx.png'.format(model))

