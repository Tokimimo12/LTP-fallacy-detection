import json
import random
import logging
from retry import retry
from transformers import pipeline, AutoConfig
from transformers import AutoTokenizer
import string

from tenacity import retry, stop_after_attempt, wait_fixed
import sys
import pandas as pd
import os
from utils import get_possible_outputs, get_possible_classes_per_category

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import argparse

random.seed(4)

def get_example(level):
    if level == "detection":
        return "We have to practice what we preach.\n Yes\n"
    elif level == "category":
        return "We have to practice what we preach.\n Fallacy of Emotion\n"
    elif level == "class":
        return "We have to practice what we preach.\n Slogans\n"
    
class FallacyDataset(Dataset):
    def __init__(self, data, mode="zero-shot", level="detection"):
        self.data = data
        self.mode = mode
        self.level = level
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.level == "class":
            previous_answer = row['pred_categories']
        else:
            previous_answer = None
        
        prompt = get_prompt(row['snippet'], self.mode, self.level, previous_answer)

        return {
            'ID': row['ID'],
            'snippet': row['snippet'],
            'prompt': prompt,
            'fallacy_detection': row['fallacy_detection'],
            'category': row['category'],
            'class': row['class'],
            'pred_detection': row.get('pred_detection', -1),
            'pred_categories': row.get('pred_categories', -1),
        }

def get_prompt(text, mode, level, previous_answer):
    _, category_labels, class_labels = get_possible_outputs()
    possible_classes_per_category = get_possible_classes_per_category()

    if level == "detection":
        answer_template = "<Yes/No>"
    elif level == "category":
        answer_template = "<Fallacy Category>"
    elif level == "class":
        answer_template = "<Specific Fallacy Type>"

    messages = [
        {"role": "system", "content": f"Your task is to simply and promptly give bare answer the next question. The answer needs to be in the following format: {answer_template}. Do not generate anything beyond this line. Do not explain or continue after the first line."}
    ]

    if mode == "one-shot":
        messages.append({"role": "user", "content": "Below is one example of how to answer the task:"})
        messages.append({"role": "user", "content": get_example(level)})
        messages.append({"role": "user", "content": "Now it's your turn:"})
    
    if level == "detection":
        {"role": "user", "content": f"Text: {text}"},
        {"role": "user", "content": "Is the text fallacious? Only answer with 'yes' or 'no'."},
    
    elif level == "category":
        messages.append({"role": "user", "content": f"Text: {text}"})
        messages.append({"role": "user", "content": f"You previously indicated that this text was a fallacy. What category of fallacy is it? You only have to answer with one of the following labels: {category_labels}."})

    elif level == "class":
        messages.append({"role": "user", "content": f"Text: {text}"})
        messages.append({"role": "user", "content": f"You previously indicated that this text was a fallacy of category '{previous_answer}'. What specific kind of fallacy is it? You only have to answer with one of the following labels: {possible_classes_per_category[previous_answer]}."})
    
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt += "\nAssistant:\n"  # hint model it's time to respon

    return prompt

def extract_answers(answer: str) -> tuple:
    from utils import get_class_to_category
    
    lines = answer.strip().splitlines()
    class_to_category = get_class_to_category()
    print(f"Extracting answers from: {lines}")
    
    try:
        # Find the index of the assistant's response
        assistant_index = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("assistant:"))
        
        # Get the line after "Assistant:"
        response_line = lines[assistant_index + 1].strip()
        print(f"Response line: {response_line}")

        # remove all punctuation
        response_line = response_line.translate(str.maketrans('', '', string.punctuation))

        # remove "<"
        if response_line.startswith("<") and response_line.endswith(">"):
            response_line = response_line[1:-1].strip()
    
        return response_line
    except (StopIteration, IndexError, ValueError):
        # Handle the case where response doesn't follow expected format
        try:
            # Try to extract the last line which might contain the answer
            last_line = lines[-1].strip()
            
            # Clean up the specific type by removing brackets if present
            if last_line.startswith("<") and last_line.endswith(">"):
                last_line = last_line[1:-1]
            
            return last_line
            
        except (IndexError, ValueError):
            # If all else fails, return default values
            return "None"
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process statements with a language model.")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "one-shot"], default="zero-shot", help="Mode of prompting: zero-shot or one-shot.")
    parser.add_argument("--model", type=str, default="mistralai", help="Model to use for generation.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing statements.")
    args = parser.parse_args()
    MODE = args.mode
    print(f"Running in {MODE} mode.")

    #GENERATiION CONFIGURATION
    generation_models = {
    "phi-4": "microsoft/phi-4",#apparently is a text generation model and does not support question-answering tasks  
    "menda": "weathermanj/Menda-3b-Optim-200" ,
    "mistralai": "mistralai/Mistral-7B-v0.1",
    "tinyllama" : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama": "meta-llama/Llama-3.2-3B", #apparently you need token for this, to do later
    "llama-instruct": "meta-llama/Llama-3.2-3B-Instruct"
    # "llama-2": "meta-llama/Llama-2-7b"
    }

    data = pd.read_csv("../data/MM_USED_fallacy/splits/test_data.csv")
    
    model_name = generation_models[args.model]
    print(f"Using model: {model_name}")

    try:
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B")
        print("Access successful!")
    except Exception as e:
        print(f"Still having issues: {e}")

    # Load tokenizer and set padding to left
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    
    # Set pad_token to eos_token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )
    #print which device the model is on
    print(f"Model is loaded on device: {generator.device}")

    possible_classes_per_category = get_possible_classes_per_category()

    for level in ["detection", "category", "class"]:
        # For collecting results
        all_results = []
        original_results = []  # To keep track of original detection results
        print(f"Processing level: {level}")

        if level == "detection":
            data = pd.read_csv("../data/MM_USED_fallacy/splits/test_data.csv")
            print(data.head())
        else:
            # Load results from previous level and filter out non-fallacious examples
            data = pd.read_csv(os.path.join('results', f"flattened_{MODE}_{args.model}.csv"))
            print(data.head())
            if level != "detection":
                # Only keep examples detected as fallacious (yes)
                data = data[data['pred_detection'].str.lower() == 'yes']
                print(f"Filtered to {len(data)} fallacious examples for category level")
            if level == "class":
                # check if pred_categories exists, if not, skip 
                if 'pred_categories' not in data.columns:
                    print("No pred_categories found in data, skipping class level processing.")
                    continue
                # For class level, we need to filter by category
                data = data[data['pred_categories'].isin(possible_classes_per_category.keys())]
                print(f"Filtered to {len(data)} examples for class level")

        # Create dataset and dataloader
        dataset = FallacyDataset(data, mode=MODE, level=level)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Process in batches
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Generate outputs for the batch
            outputs = generator(
                batch['prompt'], 
                max_new_tokens=128,
                batch_size=len(batch['prompt'])
            )
            
            # Process each result
            for i, output in enumerate(outputs):
                output = output[0]['generated_text']
                answer = extract_answers(output)
                print(answer)
                if level == "detection":
                    all_results.append({
                        'ID': batch['ID'][i],
                        'snippet': batch['snippet'][i],
                        'pred_detection': answer,
                        'fallacy_detection': batch['fallacy_detection'][i],
                        'category': batch['category'][i],
                        'class': batch['class'][i]
                    })
                elif level == "category":
                    all_results.append({
                        'ID': batch['ID'][i],
                        'snippet': batch['snippet'][i],
                        'pred_categories': answer,
                        'pred_detection': batch['pred_detection'][i],  # Keep detection from previous level
                        'fallacy_detection': batch['fallacy_detection'][i],
                        'category': batch['category'][i],
                        'class': batch['class'][i]
                    })
                elif level == "class":
                    print("appending class level results")
                    all_results.append({
                        'ID': batch['ID'][i],
                        'snippet': batch['snippet'][i],
                        'pred_classes': answer,
                        'pred_categories': batch['pred_categories'][i],  # Keep category from previous level
                        'pred_detection': batch['pred_detection'][i],  # Keep detection from previous level
                        'fallacy_detection': batch['fallacy_detection'][i],
                        'category': batch['category'][i],
                        'class': batch['class'][i]
                    })

            # Convert results to DataFrame after each level
            results_df = pd.DataFrame(all_results)
            # print length of results_df
            print(f"Results after {level} level: {len(results_df)} examples")
            
            # Save complete results after each level
            results_df.to_csv(os.path.join('results', f"flattened_{MODE}_{args.model}.csv"), index=False)
            print(f"Results saved to {os.path.join('results', f'flattened_{MODE}_{args.model}.csv')}")
            
            # Don't clear all_results after detection level since you need to keep those results
            if level == "detection":
                original_results = all_results.copy()
            else:
                # For category and class levels, make sure we keep all the detection results
                # by combining with original detection results for non-fallacious examples
                fallacious_indices = set([item['ID'] for item in all_results])
                non_fallacious = [item for item in original_results if item['ID'] not in fallacious_indices]
                all_results.extend(non_fallacious)
        
                    
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Convert any tensor values to Python native types
    for column in results_df.columns:
        if results_df[column].apply(lambda x: hasattr(x, 'item')).any():
            results_df[column] = results_df[column].apply(lambda x: x.item() if hasattr(x, 'item') else x)

    results_df.to_csv(os.path.join('results', f"flattened_{MODE}_{args.model}.csv"), index=False)
    print(f"Results saved to {os.path.join('results', f'flattened_{MODE}_{args.model}.csv')}")