import sys
import os
import torch
import csv
from torch.utils.data import DataLoader
import argparse


from Model.model import get_model, get_tokenizer
from data.MM_USED_fallacy.MM_Dataset import MM_Dataset
from Evaluation.HierarchicalEvaluator import HierarchicalEvaluator
from Utils.utils import *

def tokenize(data_batch, tokenizer, max_length=50):
    tokenized = tokenizer(data_batch, max_length = max_length, truncation=True, padding = "longest", return_tensors="pt")

    return [tokenized]

def get_saved_model_path(model_filename):
    model_path = os.path.join("Saved Models", model_filename)
    print("Model path: ", model_path)

    return model_path

def eval(test_loader, model_filename, tokenizer, head_type="MTL 6"):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(device, model_name='Roberta', head_type=head_type)

    saved_model_path = get_saved_model_path(model_filename)
    state_dict = torch.load(saved_model_path, map_location=device)
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)


    model.eval()

    if head_type == "MTL 6":
        evaluator = HierarchicalEvaluator(num_classes=6, head_type=head_type)
    elif head_type == "MTL 2":
        evaluator = HierarchicalEvaluator(num_classes=6, head_type=head_type)
    elif head_type == "STL":
        evaluator = HierarchicalEvaluator(num_classes=7, head_type=head_type)  # 6 classes + 1 for non-fallacy

    with torch.no_grad():
        for snippets, labels in test_loader:
            tokenized = tokenize(list(snippets), tokenizer, max_length=50)[0]
            ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            output = model(ids, attention_mask)

            detection_label, group_label, classify_label = [x.to(device).long() for x in labels]

            if head_type == "MTL 6" or head_type == "MTL 2":
                detection, group, classify = output
            elif head_type == "STL":
                classify = output


            # Evaluate predictions 
            if head_type == "MTL 6" or head_type == "MTL 2":
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


    print("Test Metrics:")
    evaluator_metrics = str(evaluator)
    print(evaluator)
    avg_class_f1 = evaluator.get_avg_class_f1()
    avg_class_and_detection_f1 = evaluator.get_avg_class_and_detection_f1()

    # Save metrics to csv
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Saved Metrics'))
    csv_save_path = os.path.join(save_dir, model_filename + '_test_metrics.csv')
    with open(csv_save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Macro Avg Test Class F1', 'Macro Avg Test Class & Detection F1'])
        writer.writerow([avg_class_f1, avg_class_and_detection_f1])

    # Save best evalutor output to txt
    txt_save_path = os.path.join(save_dir, model_filename + '_test_evaluator.txt')
    with open(txt_save_path, 'w') as file:
        file.write(str(evaluator_metrics))

def load_datasets(test_snippets, test_labels, batch_size = 8, head_type = "MTL 6"):
    test_dataset = MM_Dataset(test_snippets, test_labels, head_type=head_type)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader


def eval_model(model_filename, data_filename, bert_model_name = "Roberta", head_type="MTL 6", batch_size=8):
    # Load data
    test_snippets, test_labels = get_test_data(test_file_name=data_filename)
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_name=bert_model_name)

    # Get dataloaders
    test_loader = load_datasets(
        test_snippets, test_labels,
        batch_size=batch_size,
        head_type=head_type,
    )

    eval(test_loader, model_filename, tokenizer, head_type=head_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model on fallacy detection and classification')
    parser.add_argument('--model_filename', type=str, default='Roberta_MTL 6_Augmentation_None_best.pth', help='Filename of the model to evaluate')
    parser.add_argument('--test_filename', type=str, help='Filename of test data file')
    parser.add_argument('--batch_size', type=int)

    args = parser.parse_args()


    eval_model(model_filename=args.model_filename, data_filename=args.datafilename, batch_size=args.batch_size)