import sys
import os
import torch
import csv
from hierarchicalsoftmax import HierarchicalSoftmaxLoss
from torch.utils.data import DataLoader
import argparse


from model import get_model, get_tokenizer, HTCModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.MM_USED_fallacy.MM_Dataset import MM_Dataset, HTC_MM_Dataset
from Evaluation.HierarchicalEvaluator import HierarchicalEvaluator
from Utils.utils import *


def tokenize(data_batch, tokenizer, max_length=50):
    tokenized = tokenizer(data_batch, max_length = max_length, truncation=True, padding = "longest", return_tensors="pt")

    return [tokenized]

def get_saved_model_path(bert_model_name, head_type, augment, job_id):
    saved_model_name = bert_model_name + "_" + head_type + "_Augmentation:" + augment + "_best.pth"

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    ltp_dir = os.path.dirname(current_script_dir)
    trained_models_dir = os.path.join(ltp_dir, f"Trained_Models/{job_id}/Saved_Models")
    model_path = os.path.join(trained_models_dir, str(saved_model_name))
    print("Model path: ", model_path)

    return model_path

def eval(job_id, test_loader, bert_model_name, tokenizer, head_type="MTL 6", htc=False, augment=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(device, model_name=bert_model_name, head_type=head_type, htc=htc, root=get_tree() if htc else None)

    saved_model_path = get_saved_model_path(bert_model_name, head_type, augment, job_id)
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


    print("\Test Metrics:")
    evaluator_metrics = str(evaluator)
    print(evaluator)
    avg_class_f1 = evaluator.get_avg_class_f1()
    avg_class_and_detection_f1 = evaluator.get_avg_class_and_detection_f1()
    rev_precision, rev_recall, rev_f1 = evaluator.compute_reversed_f1_for_class_6()

    # Save metrics to csv
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Saved_Test_Metrics'))
    model_file_name = bert_model_name + "_" + ("HTC" if htc else head_type) + "_Augmentation:" + augment
    csv_save_path = os.path.join(save_dir, model_file_name + '_test_metrics.csv')
    with open(csv_save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Avg Test Class F1', 'Avg Test Class & Detection F1', 'Reversed Precision', 'Reversed Recall', 'Reversed F1'])
        writer.writerow([avg_class_f1, avg_class_and_detection_f1, rev_precision, rev_recall, rev_f1])

    # Save best evalutor output to txt
    txt_save_path = os.path.join(save_dir, model_file_name + '_test_evaluator.txt')
    with open(txt_save_path, 'w') as file:
        file.write(str(evaluator_metrics))

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


def eval_model(job_id, bert_model_name = "DistilBert", head_type="MTL 6", augment="None", num_epochs=5, batch_size=8):
    # Load data
    htc = False
    if head_type == "HTC":
        htc = True
        head_type = "STL" # Change model head type to STL since HTC was just a placeholder till here
        train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, unique_classes = get_data(augment=augment, htc=True)
        root = get_tree()
        # loss_weights = None
    else:
        train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels = get_data(augment=augment)
        # loss_weights = get_loss_class_weighting(train_labels, head_type=head_type)
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_name=bert_model_name)

    # Get dataloaders
    _, _, test_loader = load_datasets(
        train_snippets, train_labels, 
        val_snippets, val_labels, 
        test_snippets, test_labels,
        batch_size=batch_size,
        head_type=head_type,
        htc = htc,
        data_labels=unique_classes if htc else None,
        root=root if htc else None
    )

    eval(job_id, test_loader, bert_model_name, tokenizer, head_type=head_type, htc=htc, augment=augment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot metrics for fallacy detection')
    parser.add_argument('--job_id', type=str, help='JOB_ID of the folder where the trained models are stored')
    parser.add_argument('--head_type_list', type=list, default=["MTL 6", "HTC", "STL"], help='List of heads types to test')
    parser.add_argument('--augment_list', type=list, default=["None", "EDA", "LLM", "Undersample"], help='List of augmentation methods to test')
    parser.add_argument('--bert_model_name_list', type=list, default=["DistilBert", "Bert", "Roberta"], help='List of bert models to test')
    args = parser.parse_args()

    # Example usage
    head_type = "MTL 6"  # Set to "STL" for single task, "MTL 2" for 2 classes in final layer, and "HTC" for hierarchical softmax
    augment = "LLM"  # Set to "None" for no augmentation, "EDA" for EDA augmentation or "LLM" for LLM generated augmentation, "LLM+EDA" for both, and "Undersample" to undersample non-fallacy
    num_epochs = 20  # Adjust as needed
    batch_size = 32  # Adjust as needed

    for head_type in args.head_type_list:
        for augment in args.augment_list:
            print("Prediction Head Type: ", head_type, "Augmentation Type:", augment)
            for bert_model_name in args.bert_model_name_list:
                print(f"################ Testing with {bert_model_name} model...")
                eval_model(args.job_id, bert_model_name, head_type=head_type, augment=augment, num_epochs=num_epochs, batch_size=batch_size)