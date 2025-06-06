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
import pickle
from hierarchicalsoftmax import HierarchicalSoftmaxLoss

from model import get_model, get_tokenizer, HTCModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Augment.eda_augmentation import eda_augmentation
from data.MM_USED_fallacy.MM_Dataset import MM_Dataset, HTC_MM_Dataset
from data.MM_USED_fallacy.data_analysis import plot_fallacy_detection_distribution, plot_category_distribution, plot_class_distribution
from Evaluation.HierarchicalEvaluator import HierarchicalEvaluator
from HTC_utils import get_tree, get_index_dicts, get_tree_dicts, post_process_predictions

nltk.download('wordnet')

def plot_losses(train_losses, val_losses, model_name):
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
    
    save_path = os.path.join(save_dir, str(model_name) + "_loss_plot.png")
    plt.savefig(save_path)


def tokenize(data_batch, tokenizer, max_length=50):
    tokenized = tokenizer(data_batch, max_length = max_length, truncation=True, padding = "longest", return_tensors="pt")

    return [tokenized]

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
        # classify_weights = torch.tensor([0.27, 1.02, 1.23, 4.59, 5.65, 7.35], device=device)
        # classify_criterion = nn.CrossEntropyLoss(weight=classify_weights)
        classify_criterion = nn.CrossEntropyLoss()


        return classify_criterion



def train(train_loader, val_loader, bert_model_name, tokenizer, num_epochs=20, mtl=True, htc=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(device, model_name=bert_model_name, mtl=mtl, htc=htc, root=get_tree() if htc else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    if htc:
        loss_criterion = HierarchicalSoftmaxLoss(root=get_tree())
    else:
        loss_criterion = get_loss_function(device, mtl=mtl)

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

        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"\n→ Avg Train Loss: {avg_train_loss:.4f}")

        criterions = {
            "detection": loss_criterion[0] if mtl else None,
            "group": loss_criterion[1] if mtl else None,
            "classify": loss_criterion[2] if mtl else loss_criterion
        }
        val_loss = validate(model, val_loader, criterions, tokenizer, device, mtl=mtl, htc=htc)
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

    plot_losses(train_losses, val_losses, bert_model_name)

    # save the model
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Saved_Models'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multi_task_distilbert.pth")
    torch.save(model.state_dict(), save_path)



def validate(model, val_loader, criterions, tokenizer, device, mtl=True, htc=False):
    model.eval()
    total_loss = 0
    count = 0

    detection_criterion = criterions["detection"]
    group_criterion = criterions["group"]
    classify_criterion = criterions["classify"]

    if mtl:
        evaluator = HierarchicalEvaluator()
    else:
        evaluator = HierarchicalEvaluator(num_classes=7)  # 6 classes + 1 for non-fallacy

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
                combined_loss = detection_loss + mask * group_loss + mask * classify_loss 
                batch_loss = torch.mean(combined_loss)
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
                if htc:
                    output = output.result
                classify_preds = torch.argmax(output, dim=1).cpu().tolist()
                classify_gt = classify_label.cpu().tolist()

                detection_preds = [1] * len(classify_preds)
                group_preds = [1] * len(classify_preds)
                detection_gt = [1] * len(classify_preds)
                group_gt = [1] * len(classify_preds)
                if htc:
                    classify_preds = post_process_predictions(classify_preds)
                    classify_gt = post_process_predictions(classify_gt)

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

def load_datasets(train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, batch_size = 8, mtl = True, htc=False, root=None, data_labels= None):
    if htc:
        name_to_node_id, _ = get_tree_dicts(root, data_labels)
        train_dataset = HTC_MM_Dataset(train_snippets, train_labels, name_to_node_id)
        val_dataset = HTC_MM_Dataset(val_snippets, val_labels, name_to_node_id)
        test_dataset = HTC_MM_Dataset(test_snippets, test_labels, name_to_node_id)
    else:
        train_dataset = MM_Dataset(train_snippets, train_labels, mtl=mtl)
        val_dataset = MM_Dataset(val_snippets, val_labels, mtl=mtl)
        test_dataset = MM_Dataset(test_snippets, test_labels, mtl=mtl)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def get_data(augment, htc=False):
    data = pd.read_csv("../data/MM_USED_fallacy/full_data_processed.csv")
    # only use first 100 samples
    data = data.head(100)

    if htc:
        class_to_name, category_to_name, detection_to_name = get_index_dicts()
        data.loc[data['fallacy_detection'] == 0, 'class'] = -1
        data["class"] = data["class"].map(class_to_name)
        data["category"] = data["category"].map(category_to_name)
        data["fallacy_detection"] = data["fallacy_detection"].map(detection_to_name)
        unique_classes = data["class"].unique()

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

    if htc:
        return train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, unique_classes
    else:
        return train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels


def train_model(bert_model_name = "DistilBert", mtl=True, augment=True, num_epochs=5, batch_size=8, htc=False):
    # Load data
    if htc:
        train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, unique_classes = get_data(augment=augment, htc=htc)
        root = get_tree()
    else:
        train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels = get_data(augment=augment)
    
    tokenizer = get_tokenizer(model_name=bert_model_name)

    train_loader, val_loader, test_loader = load_datasets(
        train_snippets, train_labels, 
        val_snippets, val_labels, 
        test_snippets, test_labels,
        batch_size=batch_size,
        mtl=mtl,
        htc = htc,
        data_labels=unique_classes if htc else None,
        root=root if htc else None
    )
    train(train_loader, val_loader, bert_model_name, tokenizer, num_epochs=num_epochs, mtl=mtl, htc=htc)

if __name__ == "__main__":
    # Example usage
    mtl = False  # Set to False for single-task learning
    augment = False  # Set to False if you don't want to use EDA augmentation
    num_epochs = 10  # Adjust as needed
    batch_size = 16  # Adjust as needed
    htc=True

    for bert_model_name in ["DistilBert", "Bert", "Roberta"]:
        print(f"Training with {bert_model_name} model...")
        train_model(bert_model_name, mtl=mtl, augment=augment, num_epochs=num_epochs, batch_size=batch_size, htc=htc)

    