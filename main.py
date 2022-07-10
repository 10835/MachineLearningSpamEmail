from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange

rd = 2
dataset_name = 'spambase'

print(f'Fetching data from openml "{dataset_name}"...')

datasets = fetch_openml(dataset_name, return_X_y=True)

X, y = datasets;
y = pd.to_numeric(y)

selected_item = randrange(0, len(y))
m1 = X.iloc[selected_item]
m2 = (m1 != 0)
words_frequency = m1.loc[m2].iloc[:-3]
info = m1.loc[m2].iloc[-3:]

print('Succeeded. Splitting train test...')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rd)

print('Succeeded. Fitting model...')

# model = MLPClassifier(
    # alpha=0.05,
    # hidden_layer_sizes=(30),
    # learning_rate_init=0.0012,
    # random_state=rd
# )

model = RandomForestClassifier(random_state=rd)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Printing stats

print(f'Number of Features: {model.n_features_in_}')
print(f'AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,-1])}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred, digits=4, target_names=('No Spam', 'Yes Spam')))
print(end='', flush=True)

# Plotting

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,-1])

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=('No Spam', 'Yes Spam')).plot()

plt.figure()
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.figure()
words_frequency.plot(kind='bar', title=str(selected_item), color=('green' if y.iloc[selected_item] == 1 else 'red'))
plt.show()