from mamkit.data.datasets import MMUSEDFallacy, MMUSED, UKDebates, MArg, InputMode

import pandas as pd
import logging
from pathlib import Path


# Function from the MAMKIT website "https://nlp-unibo.github.io/mm-argfallacy/2025/"
# This takes quite some time since it has to download the audio even tho we dont use it
def download_data(task_name = 'afd'):
    base_data_path = Path(__file__).parent.resolve().joinpath('data/data_'+task_name)
    print("Base data path: ", base_data_path)

    # MM-USED-fallacy dataset
    mm_used_fallacy_loader = MMUSEDFallacy(
        task_name=task_name, # Choose between 'afc' or 'afd'               
        input_mode=InputMode.TEXT_ONLY, # Choose between TEXT_ONLY, AUDIO_ONLY, or TEXT_AUDIO
        base_data_path=base_data_path
    )

    return mm_used_fallacy_loader

def data_to_clean_csv(task = "detect"): # Task name either detect or classify
    task_name = "afd" if task == "detect" else "afc"

    data_obj = download_data(task_name=task_name)

    df = pd.DataFrame(data_obj.data)
    if task == "detect":
        df_filtered = df[['sentence', 'label']] # Only keep the sentence and label (if fallacy or not)
    else:
        df_filtered = df[['snippet', 'fallacy']] # Only keep snippet (of text) and label (of fallacy type)
        # Appeal to Emotion = 0, Appeal to Authority = 1, Ad Hominem = 2, False Cause = 3, Slippery slope = 4, Slogans = 5
        
    df_filtered = df_filtered.dropna() # Remove rows with empty values
    df_filtered.to_csv("data/" + task_name + '_data_processed.csv')





data_to_clean_csv(task="classify")




