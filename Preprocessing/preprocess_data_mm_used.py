import pandas as pd


def fallacy_to_category():
    fallacies = pd.read_csv("data/afc_data_processed.csv") 

    # Mappings for fallacy types to categories
    fallacy_to_category = {
        0: 0,  # Appeal to Emotion -> Fallacy of Emotion
        1: 1,  # Appeal to Authority -> Fallacy of Credibility
        2: 1,  # Ad Hominem -> Fallacy of Credibility
        3: 2,  # False Cause -> Fallacy of Logic
        4: 2,  # Slippery Slope -> Fallacy of Logic
        5: 0   # Slogans -> Fallacy of Emotion
    }

    if 'fallacy' in fallacies.columns:
        fallacies.rename(columns={'fallacy': 'label'}, inplace=True)

    
    fallacies['category'] = fallacies['label'].map(fallacy_to_category)
    fallacies[['snippet', 'category']].to_csv("data/category_data_processed.csv", index=False)

fallacy_to_category()
