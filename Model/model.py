import torch.nn as nn
from transformers import DistilBertModel, BertModel, RobertaModel, DistilBertTokenizer, BertTokenizer, RobertaTokenizer


class MultiTaskModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.bert = base_model
        
        self.detection_head = nn.Linear(in_features=hidden_size, out_features=2)
        self.group_head = nn.Linear(in_features=hidden_size, out_features=3)
        self.classify_head = nn.Linear(in_features=hidden_size, out_features=6)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS]-like token for BERT like models

        logits_detection = self.detection_head(pooled_output)
        logits_group = self.group_head(pooled_output)
        logits_classify = self.classify_head(pooled_output)

        return [logits_detection, logits_group, logits_classify]
    
class SingleTaskModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.bert = base_model
        
        self.classify_head = nn.Linear(in_features=hidden_size, out_features=7)  # +1 for the "non-fallacy" class

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS]-like token for BERT like models

        logits_classify = self.classify_head(pooled_output)

        return logits_classify
    
def get_model(device, model_name = "DistilBert", mtl=True):
    if model_name == "DistilBert":
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif model_name == "Bert":
        bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif model_name == "Roberta":
        bert_model = RobertaModel.from_pretrained("roberta-base")
    hidden_size = bert_model.config.hidden_size
    print("Hidden Size: ", hidden_size)

    if mtl:
        main_model = MultiTaskModel(bert_model, hidden_size)
    else:
        main_model = SingleTaskModel(bert_model, hidden_size)

    main_model.to(device)

    return main_model

def get_tokenizer(model_name = "DistilBert"):
    if model_name == "DistilBert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    elif model_name == "Bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "Roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    return tokenizer


