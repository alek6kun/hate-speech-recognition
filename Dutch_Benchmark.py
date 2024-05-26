from model import classify_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(root, 'Data', 'hatecheck-nl.csv')

ds_chinese = pd.read_csv(dataset_path, usecols=['test_case', 'label_gold'])

def compute_metrics(pred_labels, true_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

preds = []
real = []

# Iterate over the rows of the DataFrame
for i, row in ds_chinese.iterrows():
    comment_text = row['test_case']
    label = int(row['label_gold'] == 'hateful')
    preds.append(int(classify_text(comment_text)))
    real.append(label)

metrics = compute_metrics(preds, real)
print(f'Validation Metrics: {metrics}')
print(f'Number of 1s: {np.sum(real)}, Number of 0s: {len(real) - np.sum(real)}')