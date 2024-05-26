from model import classify_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(root, 'Data', 'SexComment.csv')

ds_chinese = pd.read_csv(dataset_path, usecols=['comment_text', 'label'])

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
for _, row in ds_chinese.iterrows():
    comment_text = row['comment_text']
    label = int(row['label'])
    preds.append(int(classify_text(comment_text)))
    real.append(label)
    #print(f'Predicted: {int(preds[-1])}, Real: ', real[-1])

metrics = compute_metrics(preds, real)
print(f'Validation Metrics: {metrics}')