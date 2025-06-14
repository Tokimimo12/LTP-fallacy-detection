import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Saved_Plots'))
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_fallacy_detection_distribution(df, augmented=False):
    suffix = "- augmented" if augmented else ""
    
    sns.countplot(x='fallacy_detection', hue='fallacy_detection', data=df, palette='Set2', legend=False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Distribution of Fallacy Detection{suffix}')
    plt.xlabel('Fallacy Detection')
    plt.ylabel('Count')


    plt.xticks(ticks=[0, 1], labels=['Non-Fallacious', 'Fallacious'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'fallacy_detection_distribution{suffix}.png'))
    plt.clf()


def plot_category_distribution(df, augmented=False):
    suffix = "- augmented" if augmented else ""

    filtered_df = df[df['fallacy_detection'] == 1]

    # Mapping category indices to names
    category_mapping = {
        0: "Fallacy of Emotion",
        1: "Fallacy of Credibility",
        2: "Fallacy of Logic"
    }

    sns.countplot(x='category', hue='category', data=filtered_df, palette='Set2', legend=False)


    plt.title(f'Distribution of Fallacy Categories{suffix}')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Fallacy Category')
    plt.ylabel('Count')


    plt.xticks(ticks=list(category_mapping.keys()), labels=list(category_mapping.values()), rotation=30, ha='right')

    plt.tight_layout()

    # Make sure SAVE_DIR is defined in your script
    plt.savefig(os.path.join(SAVE_DIR, f'category_distribution{suffix}.png'))
    plt.clf()


def plot_class_distribution(df, augmented=False):
    suffix = "- augmented" if augmented else ""

    # Filter dataframe to only rows where fallacy_detection == 1
    filtered_df = df[df['fallacy_detection'] == 1]

    sns.countplot(x='class', hue='class', data=filtered_df, palette='Set2', legend=False)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Distribution of Fallacy Classes {suffix}')
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
    df = pd.read_csv('full_data_processed.csv')

    # Set this to True or False depending on the data you're using
    is_augmented = False

    # print number of rows in the dataframe
    print(f"Number of rows in the dataframe: {len(df)}")

    plot_fallacy_detection_distribution(df, augmented=is_augmented)
    plot_category_distribution(df, augmented=is_augmented)
    plot_class_distribution(df, augmented=is_augmented)

    print_distribution(df, 'fallacy_detection', only_where_fallacy=False)
    print_distribution(df, 'category', only_where_fallacy=True)
    print_distribution(df, 'class', only_where_fallacy=True)

if __name__ == '__main__':
    main()
