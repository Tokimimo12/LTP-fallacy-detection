import torch
from transformers import pipeline as pipeline_llm
import pandas as pd
from tqdm import tqdm


def get_fallcy_definitions():
    fallacy_num_to_def = {
        0: "The unessential loading of the argument with emotional language to exploit the audience emotional instinct", # Appeal to Emotion
        1: "When the arguer mentions the name of an authority or a group of people who agreed with her claim either without providing any relevant evidence, or by mentioning popular non-experts, or the acceptance of the claim by the majority",  # Appeal to Authority
        2: "When the argument becomes an excessive attack on an arguer’s position",  # Ad Hominem
        3: "The misinterpretation of the correlation of two events for causation",  # False Cause
        4: "It suggests that an unlikely exaggerated outcome may follow an act",  # Slippery Slope
        5: "It is a brief and striking phrase used to provoke excitement of the audience"   # Slogans
    }

    return fallacy_num_to_def

def get_fallacy_num_to_name():
    fallacy_num_to_name = {
        0: "Appeal to Emotion", # Fallacy of Emotion
        1: "Appeal to Authority",  # Fallacy of Credibility
        2: "Ad Hominem",  # Fallacy of Credibility
        3: "False Cause",  # Fallacy of Logic
        4: "Slippery Slope",  # Fallacy of Logic
        5: "Slogans"   # Fallacy of Emotion
    }

    return fallacy_num_to_name

def idx_to_category():
    fallacy_to_category = {
        0: 0,  # Appeal to Emotion -> Fallacy of Emotion
        1: 1,  # Appeal to Authority -> Fallacy of Credibility
        2: 1,  # Ad Hominem -> Fallacy of Credibility
        3: 2,  # False Cause -> Fallacy of Logic
        4: 2,  # Slippery Slope -> Fallacy of Logic
        5: 0   # Slogans -> Fallacy of Emotion
    }

    return fallacy_to_category

def augment_data():
    # Load the model pipeline
    pipeline = pipeline_llm(
        task="text-generation",
        model="huggyllama/llama-7b",
        torch_dtype=torch.float16,
        device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Load data
    data = pd.read_csv("../data/MM_USED_fallacy/full_data_processed.csv")
    data_copy = data.copy()

    # Get fallacy definitions and mappings
    fallacy_num_to_def = get_fallcy_definitions()
    fallacy_num_to_name = get_fallacy_num_to_name()
    fallacy_to_category = idx_to_category()

    start_text = "Given this fallacious sentence of type "

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentence = row['snippet']
        f_detect = row['fallacy_detection']
        f_category = row['category']
        f_class = row['class']

        if f_detect == 1:
            for idx in range(6): # Iterate through all fallacy types
                if idx != f_category: # Skip the current category
                    # Generate text for the new fallacy type
                    input_text = f"{start_text} {fallacy_num_to_name[f_category]} : {sentence}. A modified version of type {fallacy_num_to_name[idx]} is:"
                    generated_text = pipeline(input_text, num_return_sequences=1)[0]['generated_text']
                    output_text = generated_text[len(input_text):].strip()

                    # Add the new row to the copy of the data
                    new_row = [len(data_copy), output_text, f_detect, idx, fallacy_to_category[idx]]
                    data_copy.loc[len(data_copy)] = new_row


    # Save the augmented data to a new CSV file
    data_copy.to_csv("../data/MM_USED_fallacy/augmented_data_llama7b.csv", index=False)
    
    # Return the generated text
    return generated_text


def main():

    # Load the data and augment it
    augment_data()

main()
