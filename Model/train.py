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
import matplotlib.pyplot as plt
from tqdm import tqdm  
import nltk 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Augment.eda_augmentation import eda_augmentation
from data.MM_USED_fallacy.MM_Dataset import MM_Dataset
from data.MM_USED_fallacy.data_analysis import plot_fallacy_detection_distribution, plot_category_distribution, plot_class_distribution
from Evaluation.HierarchicalEvaluator import HierarchicalEvaluator

nltk.download('wordnet')


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Saved_Plots'))
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(save_path)



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

    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []


    print(f"Using device: {device}")


    for epoch in range(num_epochs):
        # print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (snippets, labels) in tqdm(enumerate(train_loader)):
            batch_count += 1
            tokenized = tokenize(list(snippets), max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            # Run model
            detection, group, classify = model.forward(ids, attention_mask)

            # Caluclate loss for each head
            processed_labels = []
            for x in labels:
                if x.dtype == torch.float64:
                    x = x.float()  # Convert to float32 first
                if x.dtype != torch.long:
                    x = x.long()   # Then to long (int64)
                processed_labels.append(x.to(device))

            detection_label, group_label, classify_label = processed_labels

            detection_loss = criterion(detection, detection_label)
            group_loss = criterion(group, group_label)
            classify_loss = criterion(classify, classify_label)


            # If detection label is 0 (no fallacy), then other heads loss not added (cuz multiplied with 0)
            mask = detection_label.float()
            total_loss = detection_loss + mask * group_loss + mask * classify_loss 
            loss = torch.mean(total_loss)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            #     print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"\n→ Avg Train Loss: {avg_train_loss:.4f}")

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"→ Validation Loss: {val_loss:.4f}\n{'-'*50}")

    plot_losses(train_losses, val_losses)

    # save the model
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Saved_Models'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multi_task_distilbert.pth")
    torch.save(model.state_dict(), save_path)



def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0

    evaluator = HierarchicalEvaluator()

    with torch.no_grad():
        for snippets, labels in val_loader:
            tokenized = tokenize(list(snippets), max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            detection, group, classify = model(ids, attention_mask)

            detection_label, group_label, classify_label = [x.to(device).long() for x in labels]

            # print the ground truth and predictions
            # print("Ground Truth Labels: ", labels)
            # print("Detection Label: ", detection_label)
            # print("Group Label: ", group_label) 
            # print("Classify Label: ", classify_label)

            detection_loss = criterion(detection, detection_label)
            group_loss = criterion(group, group_label)
            classify_loss = criterion(classify, classify_label)

            total_batch_loss = detection_loss + detection_label * group_loss + detection_label * classify_loss
            batch_loss = torch.mean(total_batch_loss)

            total_loss += batch_loss.item()
            count += 1


            # Evaluate predictions 
            detection_preds = torch.argmax(detection, dim=1).cpu().tolist()
            group_preds = torch.argmax(group, dim=1).cpu().tolist()
            classify_preds = torch.argmax(classify, dim=1).cpu().tolist()

            detection_gt = detection_label.cpu().tolist()
            group_gt = group_label.cpu().tolist()
            classify_gt = classify_label.cpu().tolist()

            # Add predictions and labels to evaluator
            for det_p, grp_p, cls_p, det_g, grp_g, cls_g in zip(detection_preds, group_preds, classify_preds, detection_gt, group_gt, classify_gt):
                evaluator.add(
                    predictions=(det_p, grp_p, cls_p),
                    ground_truth=(det_g, grp_g, cls_g)
                )


    avg_loss = total_loss / count if count > 0 else 0

    print("\nValidation Metrics:")
    print(evaluator)

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



    train_data = pd.DataFrame({
        "snippet": train_snippets,
        "fallacy_detection": [label[0] for label in train_labels],
        "category": [label[1] for label in train_labels],
        "class": [label[2] for label in train_labels]
    })


    augmented_train_data = eda_augmentation(train_data)

    plot_fallacy_detection_distribution(augmented_train_data, augmented=True)
    plot_category_distribution(augmented_train_data, augmented=True)
    plot_class_distribution(augmented_train_data, augmented=True)

    train_snippets = augmented_train_data["snippet"].tolist()
    train_labels = augmented_train_data[["fallacy_detection", "category", "class"]].values.tolist()


    train_dataset = MM_Dataset(train_snippets, train_labels)
    val_dataset = MM_Dataset(val_snippets, val_labels)
    test_dataset = MM_Dataset(test_snippets, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)


    train(train_loader, val_loader, num_epochs=5)

    

# train(dummy_train, dummy_val, train_labels, val_labels)