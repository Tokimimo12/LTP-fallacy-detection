import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from hierarchicalsoftmax import SoftmaxNode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Augment.eda_augmentation import eda_augmentation

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

def plot_losses(train_losses, val_losses, model_name, head_type, avg_class_f1, augment):
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
    
    save_path = os.path.join(save_dir, str(head_type) + "_" + str(model_name) + "_Aug:" + str(augment) + "_loss_plot.png")
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

def get_data(augment, htc=False):

    # check if data is already split
    if os.path.exists("../data/MM_USED_fallacy/splits/train_data.csv"):
        print("Data already split, loading pre-split data...")
        print("Loading pre-split data...")
        train_data = pd.read_csv("../data/MM_USED_fallacy/splits/train_data.csv")
        val_data = pd.read_csv("../data/MM_USED_fallacy/splits/val_data.csv")
        test_data = pd.read_csv("../data/MM_USED_fallacy/splits/test_data.csv")
        # print length of each split
        print(f"Train data length: {len(train_data)}")
        print(f"Validation data length: {len(val_data)}")
        print(f"Test data length: {len(test_data)}")

        if htc:
            class_to_name, category_to_name, detection_to_name = get_index_dicts()
            for split in [train_data, val_data, test_data]:
                split.loc[split['fallacy_detection'] == 0, 'class'] = -1
                split["class"] = split["class"].map(class_to_name)
                split["category"] = split["category"].map(category_to_name)
                split["fallacy_detection"] = split["fallacy_detection"].map(detection_to_name)
                unique_classes = split["class"].unique()

        # Extract lists from split dataframes
        train_snippets = train_data["snippet"].tolist()
        train_labels = train_data[["fallacy_detection", "category", "class"]].values.tolist()
        val_snippets = val_data["snippet"].tolist()
        val_labels = val_data[["fallacy_detection", "category", "class"]].values.tolist()
        test_snippets = test_data["snippet"].tolist()
        test_labels = test_data[["fallacy_detection", "category", "class"]].values.tolist()
    else:
        print("Loading full data and splitting...")
        data = pd.read_csv("../data/MM_USED_fallacy/full_data_processed.csv")
        # Split the dataframe directly 
        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        print(f"Train data length: {len(train_data)}")
        print(f"Validation data length: {len(val_data)}")
        print(f"Test data length: {len(test_data)}")

        # save the csv files for the splits
        os.makedirs("../data/MM_USED_fallacy/splits", exist_ok=True)
        train_data.to_csv("../data/MM_USED_fallacy/splits/train_data.csv", index=False)
        val_data.to_csv("../data/MM_USED_fallacy/splits/val_data.csv", index=False)
        test_data.to_csv("../data/MM_USED_fallacy/splits/test_data.csv", index=False)

        if htc:
            class_to_name, category_to_name, detection_to_name = get_index_dicts()
            for split in [train_data, val_data, test_data]:
                split.loc[split['fallacy_detection'] == 0, 'class'] = -1
                split["class"] = split["class"].map(class_to_name)
                split["category"] = split["category"].map(category_to_name)
                split["fallacy_detection"] = split["fallacy_detection"].map(detection_to_name)
                unique_classes = split["class"].unique()

        # Extract lists from split dataframes
        train_snippets = train_data["snippet"].tolist()
        train_labels = train_data[["fallacy_detection", "category", "class"]].values.tolist()
        # (Same for val and test)
        val_snippets = val_data["snippet"].tolist()
        val_labels = val_data[["fallacy_detection", "category", "class"]].values.tolist()
        test_snippets = test_data["snippet"].tolist()
        test_labels = test_data[["fallacy_detection", "category", "class"]].values.tolist()


    if augment == "Undersample":
        if htc:
            # Under-sample the "no fallacy" class in the training set (and keep train set in same format)
            train_classes_no_fallacy = pd.DataFrame({
                "snippet": [snippet for snippet, label in zip(train_snippets, train_labels) if label[0] == 'no fallacy'],
                "fallacy_detection": [label[0] for label in train_labels if label[0] == 'no fallacy'],
                "category": [label[1] for label in train_labels if label[0] == 'no fallacy'],
                "class": [label[2] for label in train_labels if label[0] == 'no fallacy']
            })
            # undersample the "no fallacy" class to have the same number of samples as the smallest class
            train_classes_no_fallacy = train_classes_no_fallacy.sample(n=200, random_state=42)
            # create a new pd that has the undersampled "no fallacy" class and the rest of the training data
            train_data = pd.DataFrame({
                "snippet": [snippet for snippet, label in zip(train_snippets, train_labels) if label[0] != 'no fallacy'] + train_classes_no_fallacy["snippet"].tolist(),
                "fallacy_detection": [label[0] for label in train_labels if label[0] != 'no fallacy'] + train_classes_no_fallacy["fallacy_detection"].tolist(),
                "category": [label[1] for label in train_labels if label[0] != 'no fallacy'] + train_classes_no_fallacy["category"].tolist(),
                "class": [label[2] for label in train_labels if label[0] != 'no fallacy'] + train_classes_no_fallacy["class"].tolist()
            })
            # update the train_snippets and train_labels to the new undersampled data
            train_snippets = train_data["snippet"].tolist()
            train_labels = train_data[["fallacy_detection", "category", "class"]].values.tolist()

        else:
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

    if augment == "EDA":
        train_data = pd.DataFrame({
            "snippet": train_snippets,
            "fallacy_detection": [label[0] for label in train_labels],
            "category": [label[1] for label in train_labels],
            "class": [label[2] for label in train_labels]
        })


        augmented_train_data = eda_augmentation(train_data)

        # plot_fallacy_detection_distribution(augmented_train_data, augmented=True)
        # plot_category_distribution(augmented_train_data, augmented=True)
        # plot_class_distribution(augmented_train_data, augmented=True)

        train_snippets = augmented_train_data["snippet"].tolist()
        train_labels = augmented_train_data[["fallacy_detection", "category", "class"]].values.tolist()

    if augment == "LLM":
        llm_aug_data = pd.read_csv("../data/MM_USED_fallacy/augmented_data_zephyr_7b_beta_only_cleaned.csv")

        # Extract snippets and labels from augmented data
        aug_snippets = llm_aug_data["snippet"].tolist()
        aug_labels = llm_aug_data[["fallacy_detection", "category", "class"]].values.tolist()

        if htc:
            class_to_name, category_to_name, detection_to_name = get_index_dicts()
            # convert labels to names
            aug_labels = [[detection_to_name[label[0]], category_to_name[label[1]], class_to_name[label[2]]] for label in aug_labels]
    
        # Add augmented data to training data
        train_snippets.extend(aug_snippets)
        train_labels.extend(aug_labels)
    

    if htc:
        return train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels, unique_classes
    else:
        return train_snippets, train_labels, val_snippets, val_labels, test_snippets, test_labels
