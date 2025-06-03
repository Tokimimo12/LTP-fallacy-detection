import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
import pandas as pd
from model import MultiTaskDistilBert, SignleTaskDistilBert
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



def tokenize(data_batch, tokenizer, max_length=50):
    tokenized = tokenizer(data_batch, max_length = max_length, padding = "longest", return_tensors="pt")

    return [tokenized]

def get_model(device, mtl=True):
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    hidden_size = bert_model.config.hidden_size
    print("Hidden Size: ", hidden_size)

    if mtl:
        main_model = MultiTaskDistilBert(bert_model, hidden_size)
    else:
        main_model = SignleTaskDistilBert(bert_model, hidden_size)
    main_model.to(device)

    return main_model

def get_loss_function(device, mtl=True):
    if mtl:
        # Class imbalance weights
        detection_weights = torch.tensor([0.53, 6.84], device=device)  # Weights for detection
        group_weights = torch.tensor([0.57, 1, 4], device=device)  # Weights for group
        classify_weights = torch.tensor([0.27, 1.02, 1.23, 4.59, 5.65, 7.35], device=device)  # Weights for classify

        detection_criterion = nn.CrossEntropyLoss(weight=detection_weights, reduction='none')
        group_criterion = nn.CrossEntropyLoss(weight=group_weights, reduction='none')
        classify_criterion = nn.CrossEntropyLoss(weight=classify_weights, reduction='none')

        return detection_criterion, group_criterion, classify_criterion
    
    else:
        classify_weights = torch.tensor([0.27, 1.02, 1.23, 4.59, 5.65, 7.35], device=device)
        classify_criterion = nn.CrossEntropyLoss(weight=classify_weights)

        return classify_criterion



def train(train_loader, val_loader, tokenizer, num_epochs=20, mtl=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(device, mtl=mtl)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_criterion = get_loss_function(device, mtl=mtl)

    train_losses = []
    val_losses = []

    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        # print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (snippets, labels) in enumerate(tqdm(train_loader)):
            batch_count += 1
            tokenized = tokenize(list(snippets), tokenizer, max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            # Run model
            output = model.forward(ids, attention_mask)

            # Caluclate loss for each head
            processed_labels = []
            for x in labels:
                if x.dtype == torch.float64:
                    x = x.float()  # Convert to float32 first
                if x.dtype != torch.long:
                    x = x.long()   # Then to long (int64)
                processed_labels.append(x.to(device))

            detection_label, group_label, classify_label = processed_labels

            if mtl:
                detection, group, classify = output
                detection_criterion, group_criterion, classify_criterion = loss_criterion
                detection_loss = detection_criterion(detection, detection_label)
                group_loss = group_criterion(group, group_label)
                classify_loss = classify_criterion(classify, classify_label)

                # If detection label is 0 (no fallacy), then other heads loss not added (cuz multiplied with 0)
                mask = detection_label.float()
                total_loss = detection_loss + mask * group_loss + mask * classify_loss 
                loss = torch.mean(total_loss)
            else:
                classify = output
                classify_criterion = loss_criterion
                classify_loss = classify_criterion(classify, classify_label)
                loss = classify_loss

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            #     print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"\n→ Avg Train Loss: {avg_train_loss:.4f}")

        criterions = {
            "detection": loss_criterion[0] if mtl else None,
            "group": loss_criterion[1] if mtl else None,
            "classify": loss_criterion[2] if mtl else loss_criterion
        }
        val_loss = validate(model, val_loader, criterions, tokenizer, device, mtl=mtl)
        val_losses.append(val_loss)
        print(f"→ Validation Loss: {val_loss:.4f}\n{'-'*50}")

    plot_losses(train_losses, val_losses)

    # save the model
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Saved_Models'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multi_task_distilbert.pth")
    torch.save(model.state_dict(), save_path)



def validate(model, val_loader, criterions, tokenizer, device, mtl=True):
    model.eval()
    total_loss = 0
    count = 0

    detection_criterion = criterions["detection"]
    group_criterion = criterions["group"]
    classify_criterion = criterions["classify"]

    evaluator = HierarchicalEvaluator()

    with torch.no_grad():
        for snippets, labels in val_loader:
            tokenized = tokenize(list(snippets), tokenizer, max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            output = model(ids, attention_mask)

            detection_label, group_label, classify_label = [x.to(device).long() for x in labels]

            if mtl:
                detection, group, classify = output
                detection_loss = detection_criterion(detection, detection_label)
                group_loss = group_criterion(group, group_label)
                classify_loss = classify_criterion(classify, classify_label)

                # If detection label is 0 (no fallacy), then other heads loss not added (cuz multiplied with 0)
                mask = detection_label.float()
                total_loss = detection_loss + mask * group_loss + mask * classify_loss 
                batch_loss = torch.mean(total_loss)
            else:
                classify = output
                classify_loss = classify_criterion(classify, classify_label)
                batch_loss = classify_loss

            total_loss += batch_loss.item()
            count += 1


            # Evaluate predictions 
            if mtl:
                detection_preds = torch.argmax(detection, dim=1).cpu().tolist()
                group_preds = torch.argmax(group, dim=1).cpu().tolist()
                classify_preds = torch.argmax(classify, dim=1).cpu().tolist()

                detection_gt = detection_label.cpu().tolist()
                group_gt = group_label.cpu().tolist()
                classify_gt = classify_label.cpu().tolist()
            else:
                classify_preds = torch.argmax(output, dim=1).cpu().tolist()
                classify_gt = classify_label.cpu().tolist()

                detection_preds = [1] * len(classify_preds)
                group_preds = [1] * len(classify_preds)
                detection_gt = [1] * len(classify_preds)
                group_gt = [1] * len(classify_preds)

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

def load_datasets(train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, batch_size = 8, mtl = True):
    train_dataset = MM_Dataset(train_snippets, train_labels, mtl=mtl)
    val_dataset = MM_Dataset(val_snippets, val_labels, mtl=mtl)
    test_dataset = MM_Dataset(test_snippets, test_labels, mtl=mtl)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    return train_loader, val_loader, test_loader


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

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # MTL
    train_loader, val_loader, test_loader = load_datasets(
        train_snippets, train_labels, 
        val_snippets, val_labels, 
        test_snippets, test_labels,
        batch_size=8,
        mtl=True
    )
    train(train_loader, val_loader, tokenizer, num_epochs=5, mtl=True)

    # STL
    train_loader, val_loader, test_loader = load_datasets(
        train_snippets, train_labels, 
        val_snippets, val_labels, 
        test_snippets, test_labels,
        batch_size=8,
        mtl=False
    )
    train(train_loader, val_loader, tokenizer, num_epochs=5, mtl=False)
    