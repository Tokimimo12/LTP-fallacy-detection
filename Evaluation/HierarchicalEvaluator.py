from collections import Counter

class HierarchicalEvaluator:    
    def __init__(self):
        # Detection (binary)
        self.detection_total = 0
        self.detection_tp = 0
        self.detection_fp = 0
        self.detection_fn = 0
        self.detection_correct = 0

        # Category (multi-class)
        self.category_total = 0 
        self.category_tp = 0
        self.category_fp = 0    
        self.category_fn = 0
        self.category_correct = 0

        # Class (multi-class)
        self.class_total = 0
        self.class_tp = 0
        self.class_fp = 0
        self.class_fn = 0
        self.class_correct = 0

    def add(self, predictions, ground_truth):
        detection_pred, category_pred, class_pred = predictions
        detection_gt, category_gt, class_gt = ground_truth

        #  ------------- Detection level -------------

        self.detection_total +=1
        if detection_pred == detection_gt:
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
            self.category_total+=1
            if detection_pred == detection_gt:
                if category_pred == category_gt:
                    self.category_correct += 1
                    self.category_tp += 1
                else:
                    self.category_fp += 1
                    self.category_fn

        # -------------- Class level -------------
        if detection_gt:
            self.class_total += 1
            if detection_pred == detection_gt and category_pred == category_gt:
                if class_pred == class_gt:
                    self.class_correct += 1
                    self.class_tp += 1
                else:
                    self.class_fp += 1
                    self.class_fn += 1
        


    def calculate_metrics(self, correct, total, tp, fp, fn):
        # accuracy
        acc = correct / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        
        return acc, precision, recall, f1




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

        return{ 
            "detection": {
                "accuracy": detection[0],
                "precision": detection[1],
                "recall": detection[2],
                "f1": detection[3]
            },
            "category": {
                "accuracy": category[0],
                "precision": category[1],
                "recall": category[2],
                "f1": category[3]
            },
            "class": {
                "accuracy": class_[0],
                "precision": class_[1],
                "recall": class_[2],
                "f1": class_[3]
            }
        }
