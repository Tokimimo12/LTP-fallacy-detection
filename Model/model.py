import torch.nn as nn

class MultiTaskDistilBert(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.bert = base_model
        
        self.detection_head = nn.Linear(in_features=hidden_size, out_features=2)
        self.group_head = nn.Linear(in_features=hidden_size, out_features=3)
        self.classify_head = nn.Linear(in_features=hidden_size, out_features=6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS]-like token for DistilBERT

        logits_detection = self.detection_head(pooled_output)
        logits_group = self.group_head(pooled_output)
        logits_classify = self.classify_head(pooled_output)

        return [logits_detection, logits_group, logits_classify]
    
class SignleTaskDistilBert(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.bert = base_model
        
        self.classify_head = nn.Linear(in_features=hidden_size, out_features=6+1)  # +1 for the "non-fallacy" class

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS]-like token for DistilBERT

        logits_classify = self.classify_head(pooled_output)

        return logits_classify


