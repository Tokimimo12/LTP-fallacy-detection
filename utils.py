import pandas as pd
import numpy as np
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