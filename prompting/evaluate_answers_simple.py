import pandas as pd
from utils import get_reverse_dicts, get_possible_outputs
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Evaluation.HierarchicalEvaluator import HierarchicalEvaluator
import json
import numpy as np
import string

def post_process_pred(pred, conver_dict, level="fallacy"):
    # check if pred is a string

    if type(pred) == str:

        splits = pred.split('. ')
        answer = splits[-1:]
        answer = [a.lower() for a in answer]
        answer = answer[0]
        # remove punctuation from answer
        answer = answer.translate(str.maketrans('', '', string.punctuation))
        if level == "fallacy":
            if answer == "yes":
                return 1
            elif answer == "no":
                return 0
            else:
                return -1
        elif level == "category":
            try:
                return conver_dict[answer]
            except KeyError:
                # print(f'Unknown category: {answer}')
                return int(-1)
        elif level == "class":
            try:
                return conver_dict[answer]
            except KeyError:
                # print(f'Unknown class: {answer}')
                return -1
    else:
        # print(f"Unexpected type for pred: {type(pred)}. Expected str.")
        # print(f"Pred value: {pred}")
        return -1

def convert_tensors_to_ints(tensor):
    #  remove "tensor()" and convert to int
    if isinstance(tensor, str):
        tensor = tensor.replace("tensor(", "").replace(")", "")
    tensor = int(tensor)
    return tensor

def get_gt_data(gt_data, IDs):
    gt_data = gt_data.set_index('ID')
    detection_gt = []
    category_gt = []
    classify_gt = []

    for ID in IDs:
        if ID in gt_data.index:
            detection_gt.append(gt_data.at[ID, 'fallacy_detection'])
            category_gt.append(gt_data.at[ID, 'category'])
            classify_gt.append(gt_data.at[ID, 'class'])
        else:
            # print(f"ID {ID} not found in ground truth data.")
            detection_gt.append(-1)
            category_gt.append(-1)
            classify_gt.append(-1)

    return detection_gt, category_gt, classify_gt


def get_processed_df(data):
    ntclass, ntcat, ntfallacy = get_reverse_dicts()
    processed_df = data.copy()

    processed_df['index'] = [convert_tensors_to_ints(x) for x in processed_df['index']]

    for index, row in data.iterrows():
        detection = row['pred_detection']
        # print(f"Processing row {index}: {type(detection)}, {type(category)}, {type(spec_class)}")
        det_pred = post_process_pred(detection, ntfallacy, level="fallacy")
        if 'pred_categories' not in row or 'pred_classes' not in row:
            cat_pred = -1
            class_pred = -1
        else:
            # check if pred_categories and pred_classes exist in the row
            category = row['pred_categories']
            spec_class = row['pred_classes']
            cat_pred = post_process_pred(category, ntcat, level="category")
            class_pred = post_process_pred(spec_class, ntclass, level="class")
        # make all values ints
        det_pred = int(det_pred)
        cat_pred = int(cat_pred)
        class_pred = int(class_pred)
        # apply processing to processed_df
        processed_df.at[index, 'pred_detection'] = det_pred
        processed_df.at[index, 'pred_categories'] = cat_pred
        processed_df.at[index, 'pred_classes'] = class_pred

    return processed_df

if __name__ == "__main__":
    exps = ['zeroshot_fixed', 'oneshot_fixed', 'zeroshot_simple', 'oneshot_simple']
    
    for exp in exps:
        results_dir = os.path.join("results", exp)
        if not os.path.exists(results_dir):
            print(f"Results directory {results_dir} does not exist. Skipping evaluation for {exp}.")
            continue

        print(f"Evaluating results in directory: {results_dir}")
        for filename in os.listdir(results_dir):
            if filename.endswith(".csv"):
                model_name = filename.split('_')[1].split('.')[0]  # Extract model name from filename
                print(f"Evaluating Model: {model_name}")
                file_path = os.path.join(results_dir, filename)
                data = pd.read_csv(file_path)
                gt_data = pd.read_csv("../data/MM_USED_fallacy/splits/test_data.csv")

                # Process the DataFrame
                processed_df = get_processed_df(data)

                detection_preds = processed_df['pred_detection'].tolist()
                group_preds = processed_df['pred_categories'].tolist()
                classify_preds = processed_df['pred_classes'].tolist()
                IDs = processed_df['index'].tolist()

                detection_gt, category_gt, classify_gt = get_gt_data(gt_data, IDs)

                evaluator = HierarchicalEvaluator(num_classes=7, head_type='STL')
                group_preds = [int(x) for x in group_preds]
                classify_preds = [int(x) for x in classify_preds]
                # print(group_preds)
                # print(category_gt)

                # print(detection_preds)
                # print(detection_gt)

                # print(classify_preds)
                # print(classify_gt)

                # Add predictions and labels to evaluator
                for det_p, grp_p, cls_p, det_g, grp_g, cls_g in zip(detection_preds, group_preds, classify_preds, detection_gt, category_gt, classify_gt):
                    evaluator.add(
                        predictions=(det_p, grp_p, cls_p),
                        ground_truth=(det_g, grp_g, cls_g)
                    )

                print(evaluator)
                avg_f1 = evaluator.get_avg_class_f1()

                #log results to json
                results = {
                    "model_name": model_name,
                    "detection_accuracy": evaluator.detection_correct / evaluator.detection_total if evaluator.detection_total > 0 else 0,
                    "category_accuracy": evaluator.category_correct.sum() / evaluator.category_total.sum() if evaluator.category_total.sum() > 0 else 0,
                    "class_accuracy": evaluator.class_correct.sum() / evaluator.class_total.sum() if evaluator.class_total.sum() > 0 else 0,
                    "avg_class_f1": avg_f1,
                    "detection_f1": evaluator.detection_tp / (evaluator.detection_tp + evaluator.detection_fp + evaluator.detection_fn) if (evaluator.detection_tp + evaluator.detection_fp + evaluator.detection_fn) > 0 else 0,
                    "category_f1": np.mean([tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0 for tp, fp, fn in zip(evaluator.category_tp, evaluator.category_fp, evaluator.category_fn)]),
                    "class_f1": np.mean([tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0 for tp, fp, fn in zip(evaluator.class_tp, evaluator.class_fp, evaluator.class_fn)])
                }

                # remove csv from filename
                filename_no_csv = filename.replace('.csv', '')
                results_file = os.path.join(results_dir, f"evaluation_results_{filename_no_csv}.json")
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Results saved to {results_file}")


