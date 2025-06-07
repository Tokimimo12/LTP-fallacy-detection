import torch.nn as nn
from transformers import DistilBertModel, BertModel, RobertaModel, DistilBertTokenizer, BertTokenizer, RobertaTokenizer
from hierarchicalsoftmax import HierarchicalSoftmaxLinear


class MultiTaskModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_classes = 6):
        super().__init__()
        self.bert = base_model
        
        self.detection_head = nn.Linear(in_features=hidden_size, out_features=2)
        self.group_head = nn.Linear(in_features=hidden_size, out_features=3)
        self.classify_head = nn.Linear(in_features=hidden_size, out_features=num_classes)

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
    
def get_model(device, model_name = "DistilBert", head_type="MTL 6", htc=False, root=None):
    if model_name == "DistilBert":
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif model_name == "Bert":
        bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif model_name == "Roberta":
        bert_model = RobertaModel.from_pretrained("roberta-base")
    hidden_size = bert_model.config.hidden_size
    print("Hidden Size: ", hidden_size)

    if head_type == "MTL 6":
        main_model = MultiTaskModel(bert_model, hidden_size, num_classes=6)
    elif head_type == "MTL 2":
        main_model = MultiTaskModel(bert_model, hidden_size, num_classes=2)
    elif htc:
        print("Using Hierarchical Tree Classifier")
        main_model = HTCModel(bert_model, root=root)  # Replace `root` with the actual root node if needed
    elif head_type == "STL":
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


class HTCModel(nn.Module):
    def __init__(self, base_model, root):
        super().__init__()
        self.bert = base_model
        
        self.layer1 = nn.Linear(in_features=768, out_features=100)
        self.relu = nn.ReLU()
        self.output = HierarchicalSoftmaxLinear(in_features=100, root=root)

    def forward(self, input_ids, attention_mask):
        y = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        y = self.layer1(y.last_hidden_state[:, 0])
        y = self.relu(y)
        y = self.output(y)
        return y