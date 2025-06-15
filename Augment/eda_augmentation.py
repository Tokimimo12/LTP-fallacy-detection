from tqdm import tqdm
import nltk
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Augment.eda import eda

nltk.download('wordnet')

def eda_augmentation(data):
    """
    Perform EDA augmentation on the dataset.
    This function reads the dataset, applies EDA techniques to augment sentences,
    and saves the augmented data to a new CSV file.
    """

    # Load data
    data_copy = data.copy()

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentence = row['snippet']
        f_detect = row['fallacy_detection']
        f_category = row['category']
        f_class = row['class']

        if f_category == -1 or f_class == -1:
            print(f"Skipping row {index} due to missing category or class.")

        if len(sentence.split()) > 1:
            # only augment sentences that are fallacies (f_detect == 1)
            if f_detect == 0:
                continue
            aug_sentences = eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=10)

            for aug_sentence in aug_sentences[:-1]: # Remove the last one since it is the original
                new_row = [aug_sentence, f_detect, f_category, f_class]
                data_copy.loc[len(data_copy)] = new_row

    data_copy.to_csv("../data/MM_USED_fallacy/aug_data.csv", index=False)

    return data_copy
