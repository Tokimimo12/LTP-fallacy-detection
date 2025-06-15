import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
import os
import pandas as pd
import argparse
import numpy as np

FIG_SIZE = (10, 5)

def clean_names(bert_model_name, classification_model_name, aug_name):
    if classification_model_name == "MTL 6":
        classification_model_name = "MTL"

    if aug_name == "None":
        aug_name = "Original"
    if aug_name == "LLM":
        aug_name = "LLM Augmentation"
    if aug_name == "Undersample":
        aug_name = "Undersampling"

    return bert_model_name, classification_model_name, aug_name

def get_col_names(dir):
    if "val" in dir:
        class_and_detection_f1_name = "Avg Val Class & Detection F1"
        class_f1_name = "Avg Val Class F1"
    if "test" in dir:
        class_and_detection_f1_name = "Avg Test Class & Detection F1"
        class_f1_name = "Avg Test Class F1"
    else:
        class_and_detection_f1_name = "Avg Test Class & Detection F1"
        class_f1_name = "Avg Test Class F1"

    return class_and_detection_f1_name, class_f1_name

def get_f1_from_txt(text, classification_model_name):
    digits = []
    digit = ""
    for idx, char in enumerate(text):
        if char.isdigit():
            digit += char
        elif char == "." and digit != "":
            digit += char
        elif digit != "":
            if "." in digit: # Only append if a float (since excludes digits in text)
                digits.append(float(digit))
            digit = ""
        if idx == len(text) - 1 and digit != "":
            digits.append(float(digit))


    if classification_model_name == "MTL":
        f1 = digits[3] # F1 is fourth number
    elif classification_model_name == "STL":
        f1 = digits[-2] # F1 is second to last number

    return f1, digits


def get_metric_txt(dir="metrics"):
    metrics_list = [] # Tuple of bert model name, classification model name, augmentation name, detection F1
    file_list = os.listdir(dir)
    for file_name in file_list:
        if ".txt" in file_name:
            cleaned_filename = file_name.replace('/', '_')
            cleaned_filename = cleaned_filename.replace(':', '_')
            split_filename = cleaned_filename.split('_')
            bert_model_name = split_filename[0]
            classification_model_name = split_filename[1]
            aug_name = split_filename[3]
            # Clean names for plotting later
            bert_model_name, classification_model_name, aug_name = clean_names(bert_model_name, classification_model_name, aug_name)

            file_dir = os.path.join(dir, file_name)
            text = open(file_dir).read()
            if classification_model_name == "MTL":
                f1, digits = get_f1_from_txt(text, classification_model_name)
            else:
                csv_file_name = file_name[:-13] + "metrics.csv"
                file_dir = os.path.join(dir, csv_file_name)
                df = pd.read_csv(file_dir)
                f1 = df.loc[0, "Reversed F1"]

            metrics_list.append((bert_model_name, classification_model_name, aug_name, f1))

    return metrics_list

def get_metric_csvs(dir="metrics"):
    class_and_detection_f1_name, class_f1_name = get_col_names(dir)

    metrics_list = [] # Tuple of bert model name, classification model name, augmentation name, class & detection F1, class f1
    file_list = os.listdir(dir)
    for file_name in file_list:
        if ".csv" in file_name:
            cleaned_filename = file_name.replace('/', '_')
            cleaned_filename = cleaned_filename.replace(':', '_')
            split_filename = cleaned_filename.split('_')
            bert_model_name = split_filename[0]
            classification_model_name = split_filename[1]
            aug_name = split_filename[3]
            # Clean names for plotting later
            bert_model_name, classification_model_name, aug_name = clean_names(bert_model_name, classification_model_name, aug_name)
            
            
            file_dir = os.path.join(dir, file_name)
            df = pd.read_csv(file_dir)
            idx_max_class_and_detection_f1 = df[class_and_detection_f1_name].idxmax()
            max_class_and_detection_f1 = df.loc[idx_max_class_and_detection_f1, class_and_detection_f1_name]
            max_class_f1 = df.loc[idx_max_class_and_detection_f1, class_f1_name]

            metrics_list.append((bert_model_name, classification_model_name, aug_name, max_class_and_detection_f1, max_class_f1))

    return metrics_list

def get_class_f1s(dir, file_name):
    cleaned_filename = file_name.replace('/', '_')
    cleaned_filename = cleaned_filename.replace(':', '_')
    split_filename = cleaned_filename.split('_')
    bert_model_name = split_filename[0]
    classification_model_name = split_filename[1]
    aug_name = split_filename[3]
    # Clean names for plotting later
    bert_model_name, classification_model_name, aug_name = clean_names(bert_model_name, classification_model_name, aug_name)


    file_dir = os.path.join(dir, file_name)
    text = open(file_dir).read()
    _, digits = get_f1_from_txt(text, classification_model_name)

    if classification_model_name == "SLT":
        class_f1s = digits[-8:-2]
    else:
        class_f1s = digits[-7:-1]

    return np.array(class_f1s)


def plot_metrics(metrics_df, filename, metric_to_plot = "class_f1"):
    df = metrics_df

    # Make combo label: classifier first, then bert_model
    df['combo'] = df['classifier'] + ' | ' + df['bert_model']

    # Unique categories
    classifiers = sorted(df['classifier'].unique())

    # Specify bert_model order explicitly
    bert_models_ordered = ['Roberta', 'Bert', 'DistilBert']

    augmentation_order = ["Original", "Undersampling", "EDA", "LLM Augmentation"]
    df['augmentation'] = pd.Categorical(df['augmentation'], categories=augmentation_order, ordered=True)
    df = df.sort_values('augmentation')

    # Base colors per classifier (distinct hues)
    base_colors = sns.color_palette("tab10", n_colors=len(classifiers))
    classifier_color_map = dict(zip(classifiers, base_colors))

    # Fixed shade intensities for bert models
    bert_shades = {
        'DistilBert': 1.0,
        'Bert': 0.6,
        'Roberta': 0.3,
    }

    def shade_color(base_rgb, shade_intensity):
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        new_l = 0.4 + 0.4 * shade_intensity
        r, g, b = colorsys.hls_to_rgb(h, new_l, s)
        return (r, g, b)

    # Build palette dict keyed by combo: classifier | bert_model
    palette = {}
    for clf in classifiers:
        base = classifier_color_map[clf]
        for bert in bert_models_ordered:
            combo_key = f"{clf} | {bert}"
            palette[combo_key] = shade_color(base, bert_shades[bert])

    # Build hue_order sorted combos (classifier first, then bert_model in desired order)
    sorted_combos = []
    for clf in classifiers:
        for bert in bert_models_ordered:
            sorted_combos.append(f"{clf} | {bert}")

    plt.figure(figsize=FIG_SIZE)
    sns.barplot(
        data=df,
        x='augmentation',
        y=metric_to_plot,
        hue='combo',
        palette=palette,
        hue_order=sorted_combos,
    )

    clean_metric_str = metric_to_plot.strip().replace("_", " ").title()

    plt.title('Average ' + clean_metric_str + ' Comparing Data Used, Classification Head and Encoder Model')
    plt.ylabel('Average ' + clean_metric_str)
    plt.xlabel("Data Format")
    plt.legend(title='Classification Head | Encoder Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/" + str(filename) + "_" + str(metric_to_plot) + "_results", dpi=300, # Increased resolution
                bbox_inches='tight',
                transparent=True)
    plt.show()

def plot_class_f1s(class_f1s_df, filename):
    # Define the mapping from fallacy name to category
    # This is derived from your fallacy_num_to_name, but directly uses names as keys
    fallacy_name_to_category = {
        "Appeal to Emotion": "Fallacy of Emotion",
        "Appeal to Authority": "Fallacy of Credibility",
        "Ad Hominem": "Fallacy of Credibility",
        "False Cause": "Fallacy of Logic",
        "Slippery Slope": "Fallacy of Logic",
        "Slogans": "Fallacy of Emotion"
    }

    # 1. Melt the DataFrame to long format
    df_melted = class_f1s_df.melt(var_name='Fallacy Type', value_name='F1 Score')

    # 2. Add the 'Category' column
    df_melted['Category'] = df_melted['Fallacy Type'].map(fallacy_name_to_category)

    category_order = ["Fallacy of Credibility", "Fallacy of Emotion", "Fallacy of Logic"]
    df_melted['Category'] = pd.Categorical(df_melted['Category'], categories=category_order, ordered=True)

    # 3. Sort by Category to group similar classes together
    df_melted = df_melted.sort_values(by=['Category', 'Fallacy Type'])

    plt.figure(figsize=FIG_SIZE)

    ax = sns.barplot(
        x='Fallacy Type',
        y='F1 Score',
        data=df_melted,
        hue='Category',         # Color bars based on category
        dodge=False,            # Don't dodge bars, let sorting handle grouping
        palette='viridis' 
    )

    # Add F1 scores on top of the bars
    for idx, p in enumerate(ax.patches):
        if p.get_height() < 0.02:
            continue
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                    fontsize=9, color='black')
        
    cleaned_filename = filename.replace('/', '_')
    cleaned_filename = cleaned_filename.replace(':', '_')
    split_filename = cleaned_filename.split('_')
    bert_model_name = split_filename[0]
    classification_model_name = split_filename[1]
    aug_name = split_filename[3]
    # Clean names for plotting later
    bert_model_name, classification_model_name, aug_name = clean_names(bert_model_name, classification_model_name, aug_name)
    

    # Customize titles and labels
    plt.title('Class F1 Scores by Fallacy Type for ' + str(bert_model_name) + ' ' + str(classification_model_name) + ' With ' + aug_name + ' Data')
    plt.xlabel('Fallacy Type')
    plt.ylabel('Class F1 Score')
    plt.ylim(0, 1.0) 

    plt.xticks(rotation=45, ha='right')
    plt.yticks(fontsize=10)

    plt.legend(title='Fallacy Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("plots/" + str(bert_model_name) + '_' + str(classification_model_name) + '_' + str(aug_name) + "_Class_F1s_results", dpi=300, # Increased resolution
                bbox_inches='tight',
                transparent=True)
    plt.show()
                



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot metrics for fallacy detection')
    parser.add_argument('--data_dir', type=str, default='test metrics', help='Directory containing the metrics')
    args = parser.parse_args()

    csv_metrics_list = get_metric_csvs(dir=args.data_dir)
    df_csv = pd.DataFrame(csv_metrics_list, columns=['bert_model', 'classifier', 'augmentation', 'class_and_detection_f1', 'class_f1'])
    plot_metrics(df_csv, filename = args.data_dir, metric_to_plot='class_f1')

    text_metrics_list = get_metric_txt(dir=args.data_dir)
    df_txt = pd.DataFrame(text_metrics_list, columns=['bert_model', 'classifier', 'augmentation', 'detection_f1'])
    plot_metrics(df_txt, filename = args.data_dir, metric_to_plot='detection_f1')

    best_filename = "Roberta_MTL 6_Augmentation:Undersample_test_evaluator.txt"
    class_f1s_list = get_class_f1s(dir=args.data_dir, file_name=best_filename)
    df_f1s = pd.DataFrame([class_f1s_list], columns=['Appeal to Emotion', 'Appeal to Authority', 'Ad Hominem', 'False Cause', 'Slippery Slope', 'Slogans'])
    plot_class_f1s(df_f1s, filename = best_filename[:-4])

main()
