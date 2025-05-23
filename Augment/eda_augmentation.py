from tqdm import tqdm
import nltk
import pandas as pd

from eda import eda

nltk.download('wordnet')

# Load data
data = pd.read_csv("../data/MM_USED_fallacy/full_data_processed.csv")
data_copy = data.copy()

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    sentence = row['snippet']
    f_detect = row['fallacy_detection']
    f_category = row['category']
    f_class = row['class']

    if len(sentence.split()) > 1:
        aug_sentences = eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=5)

        for aug_sentence in aug_sentences[:-1]: # Remove the last one since it is the original
            new_row = [len(data_copy), aug_sentence, f_detect, f_category, f_class]
            data_copy.loc[len(data_copy)] = new_row

data_copy.to_csv("../data/MM_USED_fallacy/aug_data.csv", index=False)
