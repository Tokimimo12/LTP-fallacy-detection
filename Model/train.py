import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
import pandas as pd
from model import MultiTaskDistilBert
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.MM_USED_fallacy.MM_Dataset import MM_Dataset


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


def train(train_loader, val_loader, num_epochs=20):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (snippets, labels) in enumerate(train_loader):
            batch_count += 1
            tokenized = tokenize(list(snippets), max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            detection, group, classify = model(ids, attention_mask)
            detection_label, group_label, classify_label = [x.to(device).long() for x in labels]

            detection_loss = criterion(detection, detection_label)
            group_loss = criterion(group, group_label)
            classify_loss = criterion(classify, classify_label)

            total_loss = detection_loss + detection_label * group_loss + detection_label * classify_loss 
            loss = torch.mean(total_loss)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 5 batches or at the end of the epoch
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / batch_count

        print(f"\n→ Avg Train Loss: {avg_train_loss:.4f}")

        val_loss = validate(model, val_loader, criterion, device)
        print(f"→ Validation Loss: {val_loss:.4f}\n{'-'*50}")

        validate(model, val_loader, criterion, device)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for snippets, labels in val_loader:
            tokenized = tokenize(list(snippets), max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            detection, group, classify = model(ids, attention_mask)

            detection_label, group_label, classify_label = [x.to(device).long() for x in labels]

            detection_loss = criterion(detection, detection_label)
            group_loss = criterion(group, group_label)
            classify_loss = criterion(classify, classify_label)

            total_batch_loss = detection_loss + detection_label * group_loss + detection_label * classify_loss
            batch_loss = torch.mean(total_batch_loss)

            total_loss += batch_loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else 0
    return avg_loss




##################### old train function w/o dataloader ######################

        # tokenized_train_data = tokenize(train_data, max_length=4)
        # tokenized_val_data = tokenize(val_data, max_length=4)

        # for idx, (data, label) in enumerate(zip(tokenized_train_data, train_labels)):
        #     # Get input data
        #     ids = data["input_ids"].to(device)
        #     attention_mask = data["attention_mask"].to(device)

        #     # Run model
        #     detection, group, classify = model.forward(ids, attention_mask)

        #     # Caluclate loss for each head
        #     detection_label, group_label, classify_label = torch.tensor(label).to(device)
        #     detection_loss = criterion(detection, detection_label.unsqueeze(dim=0))
        #     group_loss = criterion(group, group_label.unsqueeze(dim=0))
        #     classify_loss = criterion(classify, classify_label.unsqueeze(dim=0))

        #     # If detection label is 0 (no fallacy), then other heads loss not added (cuz multiplied with 0)
        #     total_loss = detection_loss + detection_label * group_loss + detection_label * classify_loss 
        #     loss = torch.mean(total_loss)

        #     print("Loss: ", loss.item())

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()



# Load data, but for now using dummy data
# dummy_train = ["I have beans"]
# dummy_val = ["I eat beans"]
# train_labels = [[1,1,1]]
# val_labels = [[0,0,0]]


if __name__ == "__main__":
    data = pd.read_csv("../data/MM_USED_fallacy/full_data_processed.csv")

    # make the texts and labels into lists
    snippets = data["snippet"].tolist()
    labels = data[["fallacy_detection", "category", "class"]].values.tolist()

    # shuffles the entire data before splitting 
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # data splitting train/val/test (80/10/10)
    train_snippets, temp_snippets, train_labels, temp_labels = train_test_split(
        snippets, labels, test_size=0.2, random_state=42
    )
    val_snippets, test_snippets, val_labels, test_labels = train_test_split(
        temp_snippets, temp_labels, test_size=0.5, random_state=42
    )

    train_dataset = MM_Dataset(train_snippets, train_labels)
    val_dataset = MM_Dataset(val_snippets, val_labels)
    test_dataset = MM_Dataset(test_snippets, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    print(train_loader.dataset[0])

    train(train_loader, val_loader, num_epochs=20)

    

# train(dummy_train, dummy_val, train_labels, val_labels)