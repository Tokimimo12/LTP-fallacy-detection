import numpy as np
import matplotlib.pyplot as plt
import os

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
                


