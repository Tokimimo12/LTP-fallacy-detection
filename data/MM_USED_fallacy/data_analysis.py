import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Saved_Plots'))
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_fallacy_detection_distribution(df, augmented=False):
    suffix = "- augmented" if augmented else ""
    
    sns.countplot(x='fallacy_detection', hue='fallacy_detection', data=df, palette='Set3', legend=False)
    plt.title(f'Distribution of Fallacy Detection{suffix}')
    plt.xlabel('Fallacy Detected (1=True, 0=False)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fallacy_detection_distribution{suffix}.png'))
    plt.clf()

def plot_category_distribution(df, augmented=False):
    suffix = "- augmented" if augmented else ""

    filtered_df = df[df['fallacy_detection'] == 1]

    sns.countplot(x='category', data=filtered_df, palette='Set3', legend=False)

    plt.title(f'Distribution of Fallacy Categories (only when fallacy detected){suffix}')
    plt.xlabel('Fallacy Category')
    plt.ylabel('Count')

    # Mapping category indices to names
    category_mapping = {
        0: "Fallacy of Emotion",
        1: "Fallacy of Credibility",
        3: "Fallacy of Logic"
    }

    plt.xticks(ticks=list(category_mapping.keys()), labels=list(category_mapping.values()), rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'category_distribution{suffix}.png'))
    plt.clf()

def plot_class_distribution(df, augmented=False):
    suffix = "- augmented" if augmented else ""

    # Filter dataframe to only rows where fallacy_detection == 1
    filtered_df = df[df['fallacy_detection'] == 1]

    sns.countplot(x='class', data=filtered_df, palette='Set3', legend=False)

    plt.title(f'Distribution of Fallacy Classes (only when fallacy detected){suffix}')
    plt.xlabel('Fallacy Class')
    plt.ylabel('Count')

    class_mapping = {
        0: "Appeal to Emotion",
        1: "Appeal to Authority",
        2: "Ad Hominem",
        3: "False Cause",
        4: "Slippery Slope",
        5: "Slogans"
    }

    plt.xticks(ticks=list(class_mapping.keys()), labels=list(class_mapping.values()), rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'class_distribution{suffix}.png'))
    plt.clf()

def print_distribution(df, column_name, only_where_fallacy=True):
    if only_where_fallacy:
        df = df[df['fallacy_detection'] == 1]

    counts = df[column_name].value_counts(normalize=True) * 100
    print(f"\nDistribution for '{column_name}' (fallacy_detection == 1):")
    print(counts.round(2).astype(str) + " %")

def main():
    df = pd.read_csv('aug_data.csv')

    # Set this to True or False depending on the data you're using
    is_augmented = True

    plot_fallacy_detection_distribution(df, augmented=is_augmented)
    plot_category_distribution(df, augmented=is_augmented)
    plot_class_distribution(df, augmented=is_augmented)

    print_distribution(df, 'fallacy_detection', only_where_fallacy=False)
    print_distribution(df, 'category', only_where_fallacy=True)
    print_distribution(df, 'class', only_where_fallacy=True)

if __name__ == '__main__':
    main()
