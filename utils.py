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


