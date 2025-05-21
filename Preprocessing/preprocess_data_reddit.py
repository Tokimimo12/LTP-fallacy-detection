import pandas as pd
from collections import Counter
import ast

def extract_fallacy_data(file_path):

    extracted_data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip newline and split by tab
            parts = line.strip().split('\t')
        

            data_entry = {
                'id': parts[0],
                'tokens': parts[1],
                'reconstructed_text': parts[2],
                'fallacy_flag': parts[3],
                'fallacy_type': parts[4],
            }

            extracted_data.append(data_entry)
    
    return extracted_data


def get_overall_fallacy_flag(flag_list_str):
    flags = ast.literal_eval(flag_list_str)
    return "fallacy" if flags.count("fallacy") > flags.count("non_fallacy") else "non_fallacy"


def get_overall_fallacy_type(type_list_str, flag_list_str):
    types = ast.literal_eval(type_list_str)
    flags = ast.literal_eval(flag_list_str)

    if flags.count("fallacy") > flags.count("non_fallacy"):
        # Include only types for tokens marked as  "fallacy" and exclude "none"
        filtered = [t for t, f in zip(types, flags) if f == "fallacy" and t != "none"]
        if filtered:
            return Counter(filtered).most_common(1)[0][0]
    return "none"

def preprocess_dataset(file_path):
    # Load and convert to DataFrame
    data = pd.DataFrame(extract_fallacy_data(file_path))
    
    # Compute additional columns
    data["overall_fallacy_flag"] = data["fallacy_flag"].apply(get_overall_fallacy_flag)
    data["overall_fallacy_type"] = data.apply(
        lambda row: get_overall_fallacy_type(row["fallacy_type"], row["fallacy_flag"]), axis=1
    )

    data.drop(columns=["fallacy_flag", "fallacy_type"], inplace=True)
    data = data[["id", "tokens", "reconstructed_text", "overall_fallacy_flag", "overall_fallacy_type"]]
    data.to_csv(file_path.replace(".txt", "_pre_processed.csv"), index=False) 
    

if __name__ == "__main__":

    preprocess_dataset("data/RedditDataset/train.txt")
    preprocess_dataset("data/RedditDataset/dev.txt")
    preprocess_dataset("data/RedditDataset/test.txt")


