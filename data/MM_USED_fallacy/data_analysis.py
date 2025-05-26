import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Saved_Plots'))
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_fallacy_detection_distribution(df):
    sns.countplot(x='fallacy_detection', hue='fallacy_detection', data=df, palette='Set3', legend=False)
    plt.title('Distribution of Fallacy Detection')
    plt.xlabel('Fallacy Detected (1=True, 0=False)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'fallacy_detection_distribution.png'))
    plt.clf()


def plot_category_distribution(df):
    filtered_df = df[df['fallacy_detection'] == 1]
    sns.countplot(x='category', hue='category', data=filtered_df, palette='Set3', legend=False)
    plt.title('Distribution of Fallacy Categories (only when fallacy detected)')
    plt.xlabel('Fallacy Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'category_distribution.png'))
    plt.clf()


def plot_class_distribution(df):
    filtered_df = df[df['fallacy_detection'] == 1]
    sns.countplot(x='class', hue='class', data=filtered_df, palette='Set3', legend=False)
    plt.title('Distribution of Fallacy Classes (only when fallacy detected)')
    plt.xlabel('Fallacy Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'class_distribution.png'))
    plt.clf()


def print_distribution(df, column_name, only_where_fallacy=True):
    if only_where_fallacy:
        filtered_df = df[df['fallacy_detection'] == 1]

    counts = filtered_df[column_name].value_counts(normalize=True) * 100
    print(f"\nDistribution for '{column_name}' (fallacy_detection == 1):")
    print(counts.round(2).astype(str) + " %")


def main():
    df = pd.read_csv('full_data_processed.csv')

    plot_fallacy_detection_distribution(df)
    plot_category_distribution(df)
    plot_class_distribution(df)

    print_distribution(df, 'fallacy_detection', only_where_fallacy=False)
    print_distribution(df, 'category', only_where_fallacy=True)
    print_distribution(df, 'class', only_where_fallacy=True)


if __name__ == '__main__':
    main()
