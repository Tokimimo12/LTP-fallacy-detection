from collections import Counter
import numpy as np

class HierarchicalEvaluator:    
    def __init__(self, num_categories=3, num_classes=6, head_type="MTL 6"):
        self.head_type = head_type

        # Detection (binary)
        self.detection_total = 0
        self.detection_tp = 0
        self.detection_fp = 0
        self.detection_fn = 0
        self.detection_correct = 0

        # Category (multi-class)
        self.category_total = np.zeros(num_categories, dtype=int)
        self.category_tp = np.zeros(num_categories, dtype=int)
        self.category_fp = np.zeros(num_categories, dtype=int)
        self.category_fn = np.zeros(num_categories, dtype=int)
        self.category_correct = np.zeros(num_categories, dtype=int)

        # Class (multi-class)
        self.class_total = np.zeros(num_classes, dtype=int)
        self.class_tp = np.zeros(num_classes, dtype=int)
        self.class_fp = np.zeros(num_classes, dtype=int)
        self.class_fn = np.zeros(num_classes, dtype=int)
        self.class_correct = np.zeros(num_classes, dtype=int)
        self.avg_class_f1 = None

        self.avg_class_and_detection_f1 = None

    def get_avg_class_f1(self):
        return self.avg_class_f1
    
    def get_avg_class_and_detection_f1(self):
        return self.avg_class_and_detection_f1


    def add(self, predictions, ground_truth):
        detection_pred, category_pred, class_pred = predictions
        detection_gt, category_gt, class_gt = ground_truth

        #  ------------- Detection level -------------

        self.detection_total += 1
        if detection_pred == detection_gt: # Correct detection prediction
            self.detection_correct += 1
            if detection_pred:
                self.detection_tp += 1
            else:
                pass # true negative, no need to count 
        else:
            if detection_pred:
                self.detection_fp += 1
            else:
                self.detection_fn += 1

        #  -------------- Category level -------------

        if detection_gt:
            self.category_total[category_gt] += 1
            if detection_pred == detection_gt: # detection is correct
                if category_pred == category_gt:
                    self.category_correct[category_gt] += 1
                    self.category_tp[category_gt] += 1
                else:
                    self.category_fp[category_pred] += 1
                    self.category_fn[category_gt] += 1
            else: # detection is wrong
                self.category_fp[category_pred] += 1
                self.category_fn[category_gt] += 1


        # -------------- Class level -------------
        if self.head_type == "MTL 2":
            class_gt = class_gt + (2 * category_gt) # convert from 2 classes per category to 6 classes
            class_pred = class_pred + (2 * category_pred) # convert from 2 classes per category to 6 classes

        if detection_gt:
            self.class_total[class_gt] += 1
            if detection_pred == detection_gt and category_pred == category_gt:
                if class_pred == class_gt:
                    self.class_correct[class_gt] += 1
                    self.class_tp[class_gt] += 1
                else:
                    self.class_fp[class_pred] += 1
                    self.class_fn[class_gt] += 1
            else: # detection or category is wrong
                self.class_fp[class_pred] += 1
                self.class_fn[class_gt] += 1
        elif self.head_type == "STL":
            self.class_total[class_gt] += 1
            if class_pred == class_gt:
                self.class_correct[class_gt] += 1
                self.class_tp[class_gt] += 1
            else:
                self.class_fp[class_pred] += 1
                self.class_fn[class_gt] += 1


        


    def calculate_metrics(self, correct, total, tp, fp, fn):
        # check if input is a numpy array, then its binary classification
        if not isinstance(correct, np.ndarray):
            acc = correct / total if total else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
            
            return acc, precision, recall, f1
        else:
            ovr_accuracy = np.sum(correct) / np.sum(total) if np.sum(total) else 0
            per_class_metrics = np.zeros((len(correct), 4))  # [accuracy, precision, recall, f1] for each class
            for i in range(len(correct)):
                if total[i] == 0:
                    acc = 0
                    precision = 0
                    recall = 0
                    f1 = 0
                else:
                    acc = correct[i] / total[i]
                    precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0
                    recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

                per_class_metrics[i] = [acc, precision, recall, f1]

                
            return ovr_accuracy, per_class_metrics




    def evaluate(self):
        detection = self.calculate_metrics(
            self.detection_correct, 
            self.detection_total, 
            self.detection_tp, 
            self.detection_fp, 
            self.detection_fn
        )

        category = self.calculate_metrics(
            self.category_correct, 
            self.category_total, 
            self.category_tp, 
            self.category_fp, 
            self.category_fn
        )

        class_ = self.calculate_metrics(
            self.class_correct, 
            self.class_total, 
            self.class_tp, 
            self.class_fp, 
            self.class_fn
        )

        self.avg_class_f1 = np.mean(class_[1][:6, 3])
        if self.head_type == "MTL 6":
            self.avg_class_and_detection_f1 = np.mean(np.append(class_[1][:6, 3], detection[3]))
        elif self.head_type == "STL":
            self.avg_class_and_detection_f1 = np.mean((class_[1][3]))

        return{ 
            "detection": {
                "accuracy": detection[0],
                "precision": detection[1],
                "recall": detection[2],
                "f1": detection[3]
            },
            "category": {
                "overall accuracy": category[0],
                "per class": {
                    "accuracy": category[1][:, 0],
                    "precision": category[1][:, 1],
                    "recall": category[1][:, 2],
                    "f1": category[1][:, 3]
                }
            },
            "class": {
                "overall accuracy": class_[0],
                "per class": {
                    "accuracy": class_[1][:, 0],
                    "precision": class_[1][:, 1],
                    "recall": class_[1][:, 2],
                    "f1": class_[1][:, 3]
                }
            }
        }
    
    # this function makes it easier to print the results in a nice way
    def __str__(self):
        metrics = self.evaluate()
        output = []

        for section, values in metrics.items():
            output.append(f"{section.capitalize()} Metrics:")

            # First, print all scalar metrics
            for key, val in values.items():
                if key != "per class" and not isinstance(val, dict):
                    output.append(f"  {key.capitalize():<18}: {val:.4f}")

            # Print overall accuracy
            if "overall accuracy" in values:
                output.append(f"  Overall Accuracy: {values['overall accuracy']:.4f}")

            # Print per-class metrics as a table
            if "per class" in values:
                per_class = values["per class"]
                headers = ["Metric"] + [f"Class {i}" for i in range(len(next(iter(per_class.values()))))] + ["|  Avg"]
                output.append("")
                output.append("  " + "  ".join(f"{h:>10}" for h in headers))

                for metric_name, metric_values in per_class.items():
                    metric_values = np.array(metric_values)
                    avg = np.mean(metric_values)
                    line = [f"{metric_name.capitalize():>10}"] + [f"{v:.4f}" for v in metric_values] + [f"|  {avg:.4f}"]
                    output.append("  " + "  ".join(f"{val:>10}" for val in line))

            output.append("")

        return "\n".join(output).strip()
        
