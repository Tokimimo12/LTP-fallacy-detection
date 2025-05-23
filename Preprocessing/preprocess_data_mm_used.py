import pandas as pd
from itertools import islice


def fallacy_to_category():
    fallacies = pd.read_csv("../data/afc_data_processed.csv") 

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
    fallacies[['snippet', 'category']].to_csv("../data/category_data_processed.csv", index=False)




def create_full_data_file():
    # Load data
    afd = pd.read_csv("../data/MM_USED_fallacy/afd_data_processed.csv")
    afc = pd.read_csv("../data/MM_USED_fallacy/afc_data_processed.csv")
    f_cat = pd.read_csv("../data/MM_USED_fallacy/category_data_processed.csv")

  
    # Add the category and fallacy detection columns to afc
    afc.columns = ["ID", "snippet", "class"]
    afc["category"] = f_cat.iloc[:, 1]
    afc["fallacy_detection"] = 1
    afc = afc[["ID", "snippet", "fallacy_detection", "category", "class"]]

    # Filter afd where label (3rd col) == 0. We only want to add the no fallacy ones
    afd_filtered = afd[afd.iloc[:, 2] == 0]
    afd_subset = afd_filtered.iloc[:, [1, 2]].copy()
    afd_subset.columns = ["snippet", "fallacy_detection"]

    # placeholders - 0 means there is no label 
    afd_subset["category"] = 0
    afd_subset["class"] = 0
    afd_subset["ID"] = -1

    afd_subset = afd_subset[["ID", "snippet", "fallacy_detection", "category", "class"]]
    combined_df = pd.concat([afc, afd_subset], ignore_index=True)


    combined_df["ID"] = range(1, len(combined_df) + 1)
    final_df = combined_df[["ID", "snippet", "fallacy_detection", "category", "class"]]
    final_df.to_csv("../data/MM_USED_fallacy/full_data_processed.csv", index=False)


# fallacy_to_category()
create_full_data_file()
