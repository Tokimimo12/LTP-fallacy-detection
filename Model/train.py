import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertModel

from model import MultiTaskDistilBert

def tokenize(data_batch, max_length=50):
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    tokenized = tokenizer(data_batch, max_length = max_length, padding = "longest", return_tensors="pt")

    return [tokenized]

def get_model(device):
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    hidden_size = bert_model.config.hidden_size
    print("Hidden Size: ", hidden_size)

    main_model = MultiTaskDistilBert(bert_model, hidden_size)
    main_model.to(device)

    return main_model


def train(train_data, val_data, train_labels, val_labels, num_epochs=20):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu") # For mac or CUDA
    
    tokenized_train_data = tokenize(train_data, max_length=4)
    tokenized_val_data = tokenize(val_data, max_length=4)
    model = get_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(num_epochs):
        for idx, (data, label) in enumerate(zip(tokenized_train_data, train_labels)):
            # Get input data
            ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            # Run model
            detection, group, classify = model.forward(ids, attention_mask)

            # Caluclate loss for each head
            detection_label, group_label, classify_label = torch.tensor(label).to(device)
            detection_loss = criterion(detection, detection_label.unsqueeze(dim=0))
            group_loss = criterion(group, group_label.unsqueeze(dim=0))
            classify_loss = criterion(classify, classify_label.unsqueeze(dim=0))

            # If detection label is 0 (no fallacy), then other heads loss not added (cuz multiplied with 0)
            total_loss = detection_loss + detection_label * group_loss + detection_label * classify_loss 
            loss = torch.mean(total_loss)

            print("Loss: ", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



# Load data, but for now using dummy data
dummy_train = ["I have beans"]
dummy_val = ["I eat beans"]
train_labels = [[1,1,1]]
val_labels = [[0,0,0]]

train(dummy_train, dummy_val, train_labels, val_labels)