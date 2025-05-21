import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel

class MultiTaskDistilBert(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.bert = base_model
        
        self.detection_head = nn.Linear(in_features=hidden_size, out_features=2)
        self.group_head = nn.Linear(in_features=hidden_size, out_features=3)
        self.classify_head = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS]-like token for DistilBERT

        logits_detection = self.detection_head(pooled_output)
        logits_group = self.group_head(pooled_output)
        logits_classify = self.classify_head(pooled_output)

        return [logits_detection, logits_group, logits_classify]


def tokenize(data_batch, max_length=50):
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    tokenized = tokenizer(data_batch, max_length = max_length, padding = "longest", return_tensors="pt")

    return tokenized


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu") # For mac or CUDA

bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
# bert_model.to(device)
hidden_size = bert_model.config.hidden_size
print("Hidden Size: ", hidden_size)

main_model = MultiTaskDistilBert(bert_model, hidden_size)
main_model.to(device)

dummy_data = ["I have beans"]

tokenized_data = tokenize(dummy_data, max_length=4)

outputs = main_model.forward(tokenized_data["input_ids"].to(device), tokenized_data["attention_mask"].to(device))

print("Output 1: ", outputs[0], "Output 2: ", outputs[1], "Output 3: ", outputs[2])

