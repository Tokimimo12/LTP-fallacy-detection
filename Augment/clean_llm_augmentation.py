import pandas as pd
from tqdm import tqdm

AUG_START_IDX = 16793

def get_input_phrases():
    input_phrases = [
        "Given this fallacious sentence of type",
        "Return only a single sentence containing",
        "Appeal to Emotion",
        "Appeal to Authority",
        "Ad Hominem",
        "False Cause",
        "Slippery Slope",
        "Slogans",
        "Appeal to"
    ]

    return input_phrases

def main(input_file="full_data_processed.csv"):
    data = pd.read_csv(input_file)
    # only copy the rows up to AUG_START_IDX
    data_copy = data.copy().iloc[:AUG_START_IDX].reset_index(drop=True)

    input_phrases = get_input_phrases()

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        if index < AUG_START_IDX:
            continue

        text = row['snippet']

        text_no_colon = text.replace(":", ".") # Replace colons with dots to avoid confusion in sentence splitting
        text_no_colon_and_dots = text_no_colon.replace("...", ".") # Replace ellipses with dots to avoid confusion in sentence splitting
        text_no_colon_and_dots_and_newline = text_no_colon_and_dots.replace("\n", ".") # Replace newlines with spaces to avoid confusion in sentence splitting
        sentences = text_no_colon_and_dots_and_newline.split(".")

        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) == 0: # Skip empty sentences
                continue

            for char in sentence:
                if char.isalpha():
                    break
            else: # If no alphabet characters are found, skip the sentence
                continue
            
            # Check if the sentence contains any of the input phrases
            input_phrase_in_sentence = False
            for phrase in input_phrases:
                if phrase in sentence:
                    input_phrase_in_sentence = True
                    break
            if input_phrase_in_sentence == False: # If none of the input phrases are in the sentence then its good
                # Add the sentence to the data_copy
                new_row = [len(data_copy) + 1, sentence, row['fallacy_detection'], row['category'], row['class']]
                data_copy.loc[len(data_copy)] = new_row
                break

    # Save the cleaned data to a new CSV file
    output_file = input_file.replace(".csv", "_cleaned.csv")
    data_copy.to_csv(output_file, index=False)
        


main(input_file = "../data/MM_USED_fallacy/augmented_data_zephyr_7b_beta.csv")