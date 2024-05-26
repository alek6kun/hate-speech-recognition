import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AutoConfig
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss

class SMARTDeBERTaClassificationModel(torch.nn.Module):
    def __init__(self, model, weight=0.02):
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels=None):
        embedder = self.model.get_input_embeddings()
        embed = embedder(input_ids)

        def eval(embed):
            outputs = self.model(inputs_embeds=embed, attention_mask=attention_mask, labels=labels)
            return outputs.logits
        
        state = eval(embed)
        
        if labels is None:
            return state
        
        smart_loss_fn = SMARTLoss(eval_fn=eval, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
        loss = torch.nn.functional.cross_entropy(state, labels)
        loss += self.weight * smart_loss_fn(embed, state)

        return state, loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained('microsoft/mdeberta-v3-base', num_labels=2)
base_model = DebertaV2ForSequenceClassification(config).from_pretrained('microsoft/mdeberta-v3-base')
model = SMARTDeBERTaClassificationModel(base_model, weight=0.02).to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load('BALANCED.pt', map_location=device))
model.eval()

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/mdeberta-v3-base')

def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        _, predicted_class = torch.max(outputs, dim=1)
    
    return predicted_class
