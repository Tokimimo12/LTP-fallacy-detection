import numpy as np
import matplotlib.pyplot as plt
import os
from hierarchicalsoftmax import SoftmaxNode

classes = ["appeal to emotion", "appeal to authority", "ad hominem", "false cause", "slippery slope", "slogans", "no fallacy"]


def get_data_imbalance(train_labels, head_type="MTL 6"):
    detection_frequencies = np.zeros(2, dtype=int)
    category_frequencies = np.zeros(3, dtype=int)
    if head_type == "MTL 6" or head_type == "MTL 2":
        class_frequencies = np.zeros(6, dtype=int)
    elif head_type == "STL":
        class_frequencies = np.zeros(7, dtype=int)

    for label in train_labels:
        detection_label, category_label, class_label = label
        if head_type == "MTL 6" or head_type == "MTL 2": # For MTL hierarchy
            detection_frequencies[int(detection_label)] += 1
            if detection_label != 0:
                category_frequencies[int(category_label)] += 1
                class_frequencies[int(class_label)] += 1
        else: # For single task model (add non-fallacy to end of class)
            if detection_label != 1:
                class_frequencies[6] += 1
            else:
                class_frequencies[int(class_label)] += 1

    return [detection_frequencies, category_frequencies, class_frequencies]

def get_loss_class_weighting(train_labels, head_type="MTL 6"):
    frequencies = get_data_imbalance(train_labels, head_type=head_type)

    detection_weights = np.zeros(2)
    category_weights = np.zeros(3)
    if head_type == "MTL 6":
        class_weights = np.zeros(6)
    elif head_type == "MTL 2":
        class_weights = np.ones(2)
    elif head_type == "STL":
        class_weights = np.zeros(7)

    weights = [detection_weights, category_weights, class_weights]

    if head_type == "MTL 6":
        for tier_idx, tier in enumerate(frequencies):
            for freq_idx, freq in enumerate(tier):
                weighting = np.sum(tier) / (freq * len(tier))
                weights[tier_idx][freq_idx] = weighting
    elif head_type == "STL":
        for freq_idx, freq in enumerate(frequencies[2]):
            weighting = np.sum(frequencies[2]) / (freq * len(frequencies[0]))
            weights[2][freq_idx] = weighting
    elif head_type == "MTL 2":
        # skip class weights
        for tier_idx, tier in enumerate(frequencies[0:2]): 
            for freq_idx, freq in enumerate(tier):
                weighting = np.sum(tier) / (freq * len(tier))
                weights[tier_idx][freq_idx] = weighting

    return weights

def plot_losses(train_losses, val_losses, model_name, head_type, avg_class_f1):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs - Avg F1 = ' + str(avg_class_f1))
    plt.legend()
    plt.grid(True)
    
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Saved_Plots'))
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, str(head_type) + "_" + str(model_name) + "_loss_plot.png")
    plt.savefig(save_path)
            
def get_possible_outputs():
    detection_labels = ["Not Fallacious", "Fallacious"]
    category_labels = ["Fallacy of Emotion", "Fallacy of Credibility", "Fallacy of Logic"]
    class_labels = ["Appeal to Emotion", "Appeal to Authority", "Ad Hominem", "False Cause", "Slippery Slope", "Slogans"]

    return detection_labels, category_labels, class_labels


def get_tree():
    root = SoftmaxNode("root")
    is_fallacy = SoftmaxNode("is fallacy", parent=root)
    no_fallacy = SoftmaxNode("no fallacy", parent=root)

    emotion = SoftmaxNode("fallacy of emotion", parent=is_fallacy)
    logic = SoftmaxNode("fallacy of logic", parent=is_fallacy)
    credibility = SoftmaxNode("fallacy of credibility", parent=is_fallacy)

    appeal_emotion = SoftmaxNode("appeal to emotion", parent=emotion)
    slogans = SoftmaxNode("slogans", parent=emotion)

    appeal_authority = SoftmaxNode("appeal to authority", parent=credibility)
    ad_hominem = SoftmaxNode("ad hominem", parent=credibility)

    false_cause = SoftmaxNode("false cause", parent=logic)
    slippery_slope = SoftmaxNode("slippery slope", parent=logic)

    root.set_indexes()
    # root.render(print=True)

    return root

def get_index_dicts():
    class_to_name = {0: "appeal to emotion",  # Appeal to Emotion -> Fallacy of Emotion
        1: "appeal to authority",  # Appeal to Authority -> Fallacy of Credibility
        2: "ad hominem",  # Ad Hominem -> Fallacy of Credibility
        3: "false cause",  # False Cause -> Fallacy of Logic
        4: "slippery slope",  # Slippery Slope -> Fallacy of Logic
        5: "slogans",   # Slogans -> Fallacy of Emotion
        -1: "no fallacy"  # No Fallacy -> No Fallacy
    }

    category_to_name = {
        0: "fallacy of emotion",
        1: "fallacy of credibility",
        2: "fallacy of logic"   
    }

    detection_to_name = {
        1: "is fallacy",
        0: "no fallacy"
    }

    return class_to_name, category_to_name, detection_to_name

def get_reverse_dicts():
    class_to_name, category_to_name, detection_to_name = get_index_dicts()
    name_to_class = {v: k for k, v in class_to_name.items()}
    name_to_category = {v: k for k, v in category_to_name.items()}
    name_to_detection = {v: k for k, v in detection_to_name.items()}

    return name_to_class, name_to_category, name_to_detection

def get_tree_dicts(root, data):
    name_to_node_id = {node.name: root.node_to_id[node] for node in root.leaves}
    # print(name_to_node_id)
    index_to_node_id = {
        i: name_to_node_id[name] for i, name in enumerate(classes)
    }

    return name_to_node_id, index_to_node_id

def post_process_predictions(predictions):
    """
    Convert model predictions to class names using the provided mapping.
    """
    _, index_to_node_id = get_tree_dicts(get_tree(), None)
    node_id_to_index = {v: k for k, v in index_to_node_id.items()}

    processed_predictions = []
    for pred in predictions:
        if isinstance(pred, list):
            # If the prediction is a list, convert each element
            processed_pred = [node_id_to_index.get(item, item) for item in pred]
        else:
            # If it's a single prediction, convert it directly
            processed_pred = node_id_to_index.get(pred, pred)
        processed_predictions.append(processed_pred)
    return processed_predictions
