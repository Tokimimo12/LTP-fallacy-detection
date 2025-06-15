import torch
from transformers import pipeline as pipeline_llm
import pandas as pd
from tqdm import tqdm
import time
import os


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

def create_prompts(data, fallacy_num_to_name, fallacy_to_category, start_text="Given this fallacious sentence of type "):
    prompts = []
    labels = []

    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentence = row['snippet']
        f_detect = row['fallacy_detection']
        f_category = row['category']
        f_class = row['class']
        
        if f_detect == 1:
            for idx in range(6): # Iterate through all fallacy types
                if idx != f_class: # Skip the current class
                    prompt = f"{start_text} {fallacy_num_to_name[f_class]} : {sentence}. Return only a single sentence containing a modified version of this sentence but of fallacy type {fallacy_num_to_name[idx]}:"
                    prompts.append(prompt)
                    new_category = fallacy_to_category[idx]  # Get the new category based on the fallacy type
                    labels.append((new_category, idx))  # Store the original category and the new category

    return prompts, labels

def augment_data(batch_size=12):
    # Load the model pipeline
    pipeline = pipeline_llm(
        task="text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.float16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=batch_size,
    )

    # Load data
    data = pd.read_csv("full_data_processed.csv")
    data_copy = data.copy()

    # Get fallacy definitions and mappings
    fallacy_num_to_def = get_fallcy_definitions()
    fallacy_num_to_name = get_fallacy_num_to_name()
    fallacy_to_category = idx_to_category()

    start_text = "Given this fallacious sentence of type "
    prompts, labels = create_prompts(data, fallacy_num_to_name, fallacy_to_category, start_text=start_text)

    
    # Process prompts in batches
    output_texts = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        output = pipeline(batch_prompts, num_return_sequences=1)
        for prompt, result in zip(batch_prompts, output):
            full = result['generated_text']
            # strip off the prompt prefix to get just the new sentence:
            clean_out = full[len(prompt):].strip()
            output_texts.append(clean_out)

    # Add the generated text to the data_copy DataFrame with the corresponding labels
    for i, text in enumerate(output_texts):
        category_label, class_label = labels[i]
        new_row = {
            'snippet': text,
            'fallacy_detection': 1,  # Mark as fallacious
            'category': category_label,
            'class': class_label
        }
        data_copy.loc[len(data_copy)] = new_row


    # Save the augmented data to a new CSV file
    os.makedirs("Saved_Data", exist_ok=True)
    data_copy.to_csv("Saved_Data/augmented_data_zephyr_7b_beta.csv", index=False)



def main():

    # Load the data and augment it
    augment_data(batch_size=12)
    
main()
