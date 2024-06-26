from datasets import load_dataset
from model import classify_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
test_texts_path = os.path.join(root, 'Split Data', 'text_test.npy')

# Load the test texts from the .npy file
test_texts = np.load(test_texts_path, allow_pickle=True)

# Load the Ethos dataset
ds_ethos = load_dataset('ethos', 'binary')

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

# Iterate over the 'train' split of the Ethos dataset
for example in ds_ethos['train']:
    if example['text'] in test_texts:
        preds.append(classify_text(example['text']))
        real.append(example['label'])

metrics = compute_metrics(preds, real)
print(f'Validation Metrics: {metrics}')