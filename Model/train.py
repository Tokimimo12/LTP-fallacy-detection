import sys
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  
import nltk 

from model import get_model, get_tokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Augment.eda_augmentation import eda_augmentation
from data.MM_USED_fallacy.MM_Dataset import MM_Dataset
from data.MM_USED_fallacy.data_analysis import plot_fallacy_detection_distribution, plot_category_distribution, plot_class_distribution
from Evaluation.HierarchicalEvaluator import HierarchicalEvaluator
from Utils.utils import *

nltk.download('wordnet')


def tokenize(data_batch, tokenizer, max_length=50):
    tokenized = tokenizer(data_batch, max_length = max_length, truncation=True, padding = "longest", return_tensors="pt")

    return [tokenized]

def get_loss_function(device, loss_weights, head_type="MTL 6"):
    detection_weights, category_weights, class_weights = loss_weights
    print("Detection Weights: ", detection_weights, " -- Category Weights: ", category_weights, " -- Class Weights: ", class_weights)

    if head_type == "MTL 6" or head_type == "MTL 2":
        detection_weights = torch.tensor(detection_weights, device=device, dtype=torch.float32)  # Weights for detection
        group_weights = torch.tensor(category_weights, device=device, dtype=torch.float32)  # Weights for group

        detection_criterion = nn.CrossEntropyLoss(weight=detection_weights, reduction='none')
        group_criterion = nn.CrossEntropyLoss(weight=group_weights, reduction='none')
        
        if head_type == "MLT 6":
            classify_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)  # Weights for classify
            classify_criterion = nn.CrossEntropyLoss(weight=classify_weights, reduction='none')

            return detection_criterion, group_criterion, classify_criterion
        else:  # MTL 2
            classify_criterion = nn.CrossEntropyLoss(reduction='none')

            return detection_criterion, group_criterion, classify_criterion
    
    elif head_type == "STL":
        classify_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
        classify_criterion = nn.CrossEntropyLoss(weight=classify_weights)

        return classify_criterion



def train(train_loader, val_loader, bert_model_name, tokenizer, loss_weights, num_epochs=20, head_type="MTL 6"):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(device, model_name=bert_model_name, head_type=head_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_criterion = get_loss_function(device, loss_weights, head_type=head_type)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    train_losses = []
    val_losses = []

    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (snippets, labels) in enumerate(train_loader):
            batch_count += 1
            tokenized = tokenize(list(snippets), tokenizer, max_length=256)[0]
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

            if head_type == "MTL 6" or head_type == "MTL 2":
                detection, group, classify = output
                detection_criterion, group_criterion, classify_criterion = loss_criterion
                detection_loss = detection_criterion(detection, detection_label)
                group_loss = group_criterion(group, group_label)
                classify_loss = classify_criterion(classify, classify_label)

                # If detection label is 0 (no fallacy), then other heads loss not added (cuz multiplied with 0)
                mask = detection_label.float()
                total_loss = detection_loss + mask * group_loss + mask * classify_loss 
                loss = torch.mean(total_loss)
            elif head_type == "STL":
                classify = output
                classify_criterion = loss_criterion
                classify_loss = classify_criterion(classify, classify_label)
                loss = classify_loss

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"\n→ Avg Train Loss: {avg_train_loss:.4f}")

        criterions = {
            "detection": loss_criterion[0] if head_type in ["MTL 6", "MTL 2"] else None,
            "group": loss_criterion[1] if head_type in ["MTL 6", "MTL 2"] else None,
            "classify": loss_criterion[2] if head_type in ["MTL 6", "MTL 2"] else loss_criterion
        }
        val_loss, avg_class_f1 = validate(model, val_loader, criterions, tokenizer, device, head_type)
        val_losses.append(val_loss)
        print(f"→ Validation Loss: {val_loss:.4f}\n{'-'*50}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    plot_losses(train_losses, val_losses, bert_model_name, head_type, avg_class_f1)

    # save the model
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Saved_Models'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multi_task_distilbert.pth")
    torch.save(model.state_dict(), save_path)



def validate(model, val_loader, criterions, tokenizer, device, head_type):
    model.eval()
    total_loss = 0
    count = 0

    detection_criterion = criterions["detection"]
    group_criterion = criterions["group"]
    classify_criterion = criterions["classify"]

    if head_type == "MTL 6":
        evaluator = HierarchicalEvaluator(num_classes=6, head_type=head_type)
    elif head_type == "MTL 2":
        evaluator = HierarchicalEvaluator(num_classes=6, head_type=head_type)
    elif head_type == "STL":
        evaluator = HierarchicalEvaluator(num_classes=7, head_type=head_type)  # 6 classes + 1 for non-fallacy

    with torch.no_grad():
        for snippets, labels in val_loader:
            tokenized = tokenize(list(snippets), tokenizer, max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            output = model(ids, attention_mask)

            detection_label, group_label, classify_label = [x.to(device).long() for x in labels]

            if head_type == "MTL 6" or head_type == "MTL 2":
                detection, group, classify = output
                detection_loss = detection_criterion(detection, detection_label)
                group_loss = group_criterion(group, group_label)
                classify_loss = classify_criterion(classify, classify_label)

                # If detection label is 0 (no fallacy), then other heads loss not added (cuz multiplied with 0)
                mask = detection_label.float()
                combined_loss = detection_loss + mask * group_loss + mask * classify_loss 
                batch_loss = torch.mean(combined_loss)
            elif head_type == "STL":
                classify = output
                classify_loss = classify_criterion(classify, classify_label)
                batch_loss = classify_loss

            total_loss += batch_loss.item()
            count += 1


            # Evaluate predictions 
            if head_type == "MTL 6" or head_type == "MTL 2":
                detection_preds = torch.argmax(detection, dim=1).cpu().tolist()
                group_preds = torch.argmax(group, dim=1).cpu().tolist()
                classify_preds = torch.argmax(classify, dim=1).cpu().tolist()

                detection_gt = detection_label.cpu().tolist()
                group_gt = group_label.cpu().tolist()
                classify_gt = classify_label.cpu().tolist()
            elif head_type == "STL":
                classify_preds = torch.argmax(output, dim=1).cpu().tolist()
                classify_gt = classify_label.cpu().tolist()

                # Placeholders since not predicted for STL
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
    avg_class_f1 = evaluator.get_avg_class_f1()

    return avg_loss, avg_class_f1

def load_datasets(train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, batch_size = 8, head_type = "MTL 6"):
    train_dataset = MM_Dataset(train_snippets, train_labels, head_type=head_type)
    val_dataset = MM_Dataset(val_snippets, val_labels, head_type=head_type)
    test_dataset = MM_Dataset(test_snippets, test_labels, head_type=head_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def get_data(augment, under_sample_non_fallacy = False):
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

    if under_sample_non_fallacy:
        # Under-sample the "no fallacy" class in the training set (and keep train set in same format)
        train_classes_no_fallacy = pd.DataFrame({
            "snippet": [snippet for snippet, label in zip(train_snippets, train_labels) if label[0] == 0],
            "fallacy_detection": [label[0] for label in train_labels if label[0] == 0],
            "category": [label[1] for label in train_labels if label[0] == 0],
            "class": [label[2] for label in train_labels if label[0] == 0]
        })
        # undersample the "no fallacy" class to have the same number of samples as the smallest class
        train_classes_no_fallacy = train_classes_no_fallacy.sample(n=200, random_state=42)
        # create a new pd that has the undersampled "no fallacy" class and the rest of the training data
        train_data = pd.DataFrame({
            "snippet": [snippet for snippet, label in zip(train_snippets, train_labels) if label[0] != 0] + train_classes_no_fallacy["snippet"].tolist(),
            "fallacy_detection": [label[0] for label in train_labels if label[0] != 0] + train_classes_no_fallacy["fallacy_detection"].tolist(),
            "category": [label[1] for label in train_labels if label[0] != 0] + train_classes_no_fallacy["category"].tolist(),
            "class": [label[2] for label in train_labels if label[0] != 0] + train_classes_no_fallacy["class"].tolist()
        })
        # update the train_snippets and train_labels to the new undersampled data
        train_snippets = train_data["snippet"].tolist()
        train_labels = train_data[["fallacy_detection", "category", "class"]].values.tolist()

    if augment:
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

    return train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels  


def train_model(bert_model_name = "DistilBert", head_type="MTL 6", augment=True, num_epochs=5, batch_size=8):
    # Load data
    train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels = get_data(augment=augment, under_sample_non_fallacy = True)

    loss_weights = get_loss_class_weighting(train_labels, head_type=head_type)
    
    tokenizer = get_tokenizer(model_name=bert_model_name)

    train_loader, val_loader, test_loader = load_datasets(
        train_snippets, train_labels, 
        val_snippets, val_labels, 
        test_snippets, test_labels,
        batch_size=batch_size,
        head_type=head_type
    )
    train(train_loader, val_loader, bert_model_name, tokenizer, loss_weights, num_epochs=num_epochs, head_type=head_type)

if __name__ == "__main__":
    # Example usage
    head_type = "MTL 6"  # Set to "STL" for single task and "MTL 2" for 2 classes in final layer
    augment = False  # Set to False if you don't want to use EDA augmentation
    num_epochs = 10  # Adjust as needed
    batch_size = 16  # Adjust as needed

    for head_type in ["MTL 6", "MTL 2", "STL"]:
        print("Prediction Head Type: ", head_type)
        for bert_model_name in ["DistilBert", "Bert", "Roberta"]:
            print(f"Training with {bert_model_name} model...")
            train_model(bert_model_name, head_type=head_type, augment=augment, num_epochs=num_epochs, batch_size=batch_size)
    