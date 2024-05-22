from datasets import load_dataset
from model import classify_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dataset = load_dataset('ethos', 'binary')
    
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
for example in dataset['train']:
    preds.append(classify_text(example['text']))
    real.append(example['label'])
metrics = compute_metrics(preds, real)
print(f'Validation Metrics: {metrics}')