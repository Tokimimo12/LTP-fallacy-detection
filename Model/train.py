import sys
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  
import nltk 
import csv
from hierarchicalsoftmax import HierarchicalSoftmaxLoss

from model import get_model, get_tokenizer, HTCModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.MM_USED_fallacy.MM_Dataset import MM_Dataset, HTC_MM_Dataset
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



def train(train_loader, val_loader, bert_model_name, tokenizer, loss_weights, num_epochs=20, head_type="MTL 6", htc=False, augment=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(device, model_name=bert_model_name, head_type=head_type, htc=htc, root=get_tree() if htc else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    if htc:
        loss_criterion = HierarchicalSoftmaxLoss(root=get_tree())
    else:
        loss_criterion = get_loss_function(device, loss_weights, head_type=head_type)

    # Early stopping parameters
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    best_evaluator_metrics = None

    train_losses = []
    val_losses = []
    avg_val_class_f1s = []
    avg_val_class_and_detection_f1s = []

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
        val_loss, evaluator_metrics, avg_class_f1, avg_class_and_detection_f1 = validate(model, val_loader, criterions, tokenizer, device, head_type, htc=htc)
        val_losses.append(val_loss)
        avg_val_class_f1s.append(avg_class_f1)
        avg_val_class_and_detection_f1s.append(avg_class_and_detection_f1)
        print(f"→ Validation Loss: {val_loss:.4f}\n{'-'*50}")

        # Check if this is the best model so far
        if avg_class_and_detection_f1 > best_val_f1:
            best_val_f1 = avg_class_and_detection_f1
            patience_counter = 0
            best_evaluator_metrics = evaluator_metrics

            # Save the best model
            save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Saved_Models'))
            model_save_path = os.path.join(save_dir, bert_model_name + "_" + ("HTC" if htc else head_type) + "_Augmentation:" + augment + "_best.pth")
            torch.save(model.state_dict(), model_save_path)

            print(f"New best model saved, with average class and detection F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    plot_losses(train_losses, val_losses, bert_model_name, head_type, avg_class_f1, augment)

    # save the model
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Saved_Models'))
    os.makedirs(save_dir, exist_ok=True)
    model_file_name = bert_model_name + "_" + ("HTC" if htc else head_type) + "_Augmentation:" + augment
    save_path = os.path.join(save_dir, model_file_name + "_final.pth")
    torch.save(model.state_dict(), save_path)

    # Save metrics to csv
    csv_save_path = os.path.join(save_dir, model_file_name + '_metrics.csv')
    with open(csv_save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Train Loss', 'Val Loss', 'Avg Val Class F1', 'Avg Val Class & Detection F1'])
        for t_loss, v_loss, v_class_f1, v_all_f1 in zip(train_losses, val_losses, avg_val_class_f1s, avg_val_class_and_detection_f1s):
            writer.writerow([t_loss, v_loss, v_class_f1, v_all_f1])

    # Save best evalutor output to txt
    txt_save_path = os.path.join(save_dir, model_file_name + '_best_evaluator.txt')
    with open(txt_save_path, 'w') as file:
        file.write(str(best_evaluator_metrics))



def validate(model, val_loader, criterions, tokenizer, device, head_type, htc=False):
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
                if htc:
                    output = output.result
                classify_preds = torch.argmax(output, dim=1).cpu().tolist()
                classify_gt = classify_label.cpu().tolist()

                # Placeholders since not predicted for STL
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
    evaluator_metrics = str(evaluator)
    print(evaluator)
    avg_class_f1 = evaluator.get_avg_class_f1()
    avg_class_and_detection_f1 = evaluator.get_avg_class_and_detection_f1()

    return avg_loss, evaluator_metrics, avg_class_f1, avg_class_and_detection_f1

def load_datasets(train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, batch_size = 8, head_type = "MTL 6", htc=False, root=None, data_labels= None):
    if htc:
        name_to_node_id, _ = get_tree_dicts(root, data_labels)
        train_dataset = HTC_MM_Dataset(train_snippets, train_labels, name_to_node_id)
        val_dataset = HTC_MM_Dataset(val_snippets, val_labels, name_to_node_id)
        test_dataset = HTC_MM_Dataset(test_snippets, test_labels, name_to_node_id)
    else:
        train_dataset = MM_Dataset(train_snippets, train_labels, head_type=head_type)
        val_dataset = MM_Dataset(val_snippets, val_labels, head_type=head_type)
        test_dataset = MM_Dataset(test_snippets, test_labels, head_type=head_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train_model(bert_model_name = "DistilBert", head_type="MTL 6", augment="None", num_epochs=5, batch_size=8):
    # Load data
    htc = False
    if head_type == "HTC":
        htc = True
        head_type = "STL" # Change model head type to STL since HTC was just a placeholder till here
        train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, unique_classes = get_data(augment=augment, htc=True)
        root = get_tree()
        loss_weights = None
    else:
        train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels = get_data(augment=augment)
        loss_weights = get_loss_class_weighting(train_labels, head_type=head_type)
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_name=bert_model_name)

    # Get dataloaders
    train_loader, val_loader, test_loader = load_datasets(
        train_snippets, train_labels, 
        val_snippets, val_labels, 
        test_snippets, test_labels,
        batch_size=batch_size,
        head_type=head_type,
        htc = htc,
        data_labels=unique_classes if htc else None,
        root=root if htc else None
    )

    train(train_loader, val_loader, bert_model_name, tokenizer, loss_weights, num_epochs=num_epochs, head_type=head_type, htc=htc, augment=augment)

if __name__ == "__main__":
    # Example usage
    head_type = "MTL 6"  # Set to "STL" for single task, "MTL 2" for 2 classes in final layer, and "HTC" for hierarchical softmax
    augment = "LLM"  # Set to "None" for no augmentation, "EDA" for EDA augmentation or "LLM" for LLM generated augmentation, "LLM+EDA" for both, and "Undersample" to undersample non-fallacy
    num_epochs = 20  # Adjust as needed
    batch_size = 32  # Adjust as needed

    for head_type in ["STL", "HTC", "MTL 6"]:
        for augment in ["None", "EDA", "LLM", "Undersample"]:
            print("Prediction Head Type: ", head_type, "Augmentation Type:", augment)
            for bert_model_name in ["DistilBert", "Bert", "Roberta"]:
                print(f"################ Training with {bert_model_name} model...")
                train_model(bert_model_name, head_type=head_type, augment=augment, num_epochs=num_epochs, batch_size=batch_size)

    