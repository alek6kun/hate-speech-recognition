import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers import BertTokenizer

root = os.path.dirname(os.path.abspath(__file__))

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/mdeberta-v3-base')

train_text_path = os.path.join(root, "Split Data", "text_train.npy")
train_label_path = os.path.join(root, "Split Data", "label_train.npy")
test_text_path = os.path.join(root, "Split Data", "text_test.npy")
test_label_path = os.path.join(root, "Split Data", "label_test.npy")

# Load the data from .npy files with allow_pickle=True
text_train = np.load(train_text_path, allow_pickle=True)
label_train = np.load(train_label_path, allow_pickle=True)
text_test = np.load(test_text_path, allow_pickle=True)
label_test = np.load(test_label_path, allow_pickle=True)

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, text_data, label_data, tokenizer, max_length=256):
        self.text_data = text_data
        self.label_data = label_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
      text = self.text_data[index]
      label = self.label_data[index]

      # Preprocess the text using the tokenizer
      # Convert text to a PyTorch tensor of token ids
      encoded_text = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
      )

      input_ids = encoded_text['input_ids'].squeeze(0)  # Remove the extra dimension
      attention_mask = encoded_text['attention_mask'].squeeze(0)

      # Convert label to a PyTorch tensor
      # First, convert the label data from a string to an integer
      if isinstance(label, str):
          label = int(label)  # Convert label from string to integer

      return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.int64)
        }


# Create the datasets for training and testing
train_dataset = CustomDataset(text_train, label_train, tokenizer)
test_dataset = CustomDataset(text_test, label_test, tokenizer)

# DataLoader parameters
batch_size = 16
shuffle = True

# Create the DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage of the DataLoader
# for batch_idx, batch in enumerate(train_loader):
#     print(f'Batch {batch_idx + 1}:')
#     print('Text batch:', batch['input_ids'][0])
#     print('Attention mask batch:', batch['attention_mask'][0])
#     print('Label batch:', batch['labels'][0])
#     break  # Remove this break statement to go through the whole DataLoader

class SMARTDeBERTaClassificationModel(nn.Module):

    def __init__(self, model, weight = 0.02):
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels):
        # Get initial embeddings
        embedder = self.model.get_input_embeddings()
        embed = embedder(input_ids)
        
        # Define eval function
        def eval(embed):
            outputs = self.model(inputs_embeds=embed, attention_mask=attention_mask, labels=labels)
            return outputs.logits

        # Define SMART loss
        smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
        # Compute initial (unperturbed) state
        state = eval(embed)
        # Apply classification loss
        loss = F.cross_entropy(state.view(-1, 2), labels.view(-1))
        # Apply smart loss
        loss += self.weight * smart_loss_fn(embed, state)

        return state, loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = AutoConfig.from_pretrained('microsoft/mdeberta-v3-base', num_labels=2)
base_model = DebertaV2ForSequenceClassification(config).from_pretrained('microsoft/mdeberta-v3-base')
model = SMARTDeBERTaClassificationModel(base_model, weight=0.02).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
scaler = GradScaler()
accumulation_steps = 2  # Gradient accumulation steps

def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_steps = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            _, loss = model(input_ids, attention_mask, labels)
            loss = loss / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (batch_idx + 1) % accumulation_steps == 0:
            print(f'Batch {batch_idx + 1}/{total_steps}, Batch Loss: {loss.item():.4f}')

    avg_loss = total_loss / total_steps
    print(f'Training Loss: {avg_loss:.4f}')
    return avg_loss

def compute_metrics(pred_labels, true_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs, _ = model(input_ids, attention_mask, labels)
                _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(predictions, true_labels)
    return metrics

num_epochs = 3  # Define the number of epochs

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_model(model, train_loader, optimizer, device)

    metrics = evaluate_model(model, test_loader, device)
    print(f'Validation Metrics: {metrics}')

torch.save(model.state_dict(), 'smart_deberta_classification_model.pt')