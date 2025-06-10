import json
import random
import logging
from retry import retry
from transformers import pipeline, AutoConfig
from transformers import AutoTokenizer

from tenacity import retry, stop_after_attempt, wait_fixed
import sys
import pandas as pd
import os
from utils import get_possible_outputs, get_possible_classes_per_category

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import argparse

random.seed(4)

class FallacyDataset(Dataset):
    def __init__(self, data, mode="zero-shot"):
        self.data = data
        self.mode = mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.mode == "zero-shot":
            prompt = prompt_zeroshot(row['snippet'])
        else:
            prompt = prompt_oneshot(row['snippet'])
        
        return {
            'index': row['ID'],
            'statement': row['snippet'],
            'prompt': prompt,
            'gt_detection': row['fallacy_detection'],
            'gt_category': row['category'],
            'gt_class': row['class']
        }

def prompt_zeroshot(text:str) -> str:
    _, category_labels, class_labels = get_possible_outputs()
    possible_classes_per_category = get_possible_classes_per_category()
    messages = [
        {"role": "system", "content": "Your task is to simply and promptly give bare answer the next 3 questions. The answer needs to be in the following format, each on a new line: \n1. <Yes/No>\n2. <Fallacy Category>\n3. <Specific Type>.\n Do not generate anything beyond these three lines. Do not explain or continue after the third line."},
        {"role": "user", "content": f"Text: {text}"},
        {"role": "user", "content": "1. Is the text fallacious? Only answer with 'yes' or 'no'."},
        {"role": "user", "content": f"2. What category of fallacy is it? You only have to answer with one of the following labels: {category_labels}, or ['None'] if it is not fallacious."},
        {"role": "user", 
            "content": f"3. What specific kind of fallacy is it? You only have to answer with one of the following labels. If you answered 'None' to the previous question, you can answer with 'None' here as well. If you answered 'Fallacy of Emotion' to the category question, you can answer with one of the following labels: {possible_classes_per_category['Fallacy of Emotion']}. If you answered 'Fallacy of Credibility' to the category question, you can answer with one of the following labels: {possible_classes_per_category['Fallacy of Credibility']}. If you answered 'Fallacy of Logic' to the category question, you can answer with one of the following labels: {possible_classes_per_category['Fallacy of Logic']}."},
    ]

    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt += "\nAssistant:\n"  # hint model it's time to respond

    print(prompt)
    return prompt

def prompt_oneshot(text:str) -> str:
    # example from train set, sample 466
    example = (
        "We have to practice what we preach.\n"
        "1. Yes\n"
        "2. Fallacy of Emotion\n"
        "3. Slogans\n"
    )

    _, category_labels, class_labels = get_possible_outputs()
    possible_classes_per_category = get_possible_classes_per_category()

    messages = [
        {   "role": "system",
            "content": "Your task is to simply and promptly give bare answer the next 3 questions. The answer needs to be in the following format, each on a new line: \n1. <Yes/No>\n2. <Fallacy Category>\n3. <Specific Type>.\n Do not generate anything beyond these three lines. Do not explain or continue after the third line."},
        {"role": "user", "content": "Below is one example of how to answer the task:"},
        {"role": "user", "content": example},
        {"role": "user", "content": "Now it's your turn:"},
        {"role": "user", "content": f"Text: {text}"},
        {"role": "user", "content": "1. Is the text fallacious? Only answer with 'yes' or 'no'."},
        {"role": "user", "content": f"2. What category of fallacy is it? You only have to answer with one of the following labels: {category_labels}, or ['None'] if it is not fallacious."},
        {"role": "user", 
            "content": f"3. What specific kind of fallacy is it? You only have to answer with one of the following labels. If you answered 'None' to the previous question, you can answer with 'None' here as well. If you answered 'Fallacy of Emotion' to the category question, you can answer with one of the following labels: {possible_classes_per_category['Fallacy of Emotion']}. If you answered 'Fallacy of Credibility' to the category question, you can answer with one of the following labels: {possible_classes_per_category['Fallacy of Credibility']}. If you answered 'Fallacy of Logic' to the category question, you can answer with one of the following labels: {possible_classes_per_category['Fallacy of Logic']}."},
    ]

    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt += "\nAssistant:\n"  # hint model it's time to respon
    return prompt

def extract_answers(answer: str) -> tuple:
    lines = answer.strip().splitlines()

    try:
        # Find the index of the assistant's response and save the 3 answers right after it
        assistant_index = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("assistant:"))
        next_lines = lines[assistant_index + 1 : assistant_index + 4]
        if len(next_lines) < 3:
            raise ValueError("Not enough lines returned by model.")
        fallacious, category, specific_type = [line.strip() for line in next_lines]
        return fallacious, category, specific_type
    except (StopIteration, IndexError, ValueError):
        # otherwise just save the last 3 lines of your response
        lines = answer.strip().splitlines()[-3:]  # last 3 lines expected
        fallacious = lines[0].strip()
        category = lines[1].strip()
        specific_type = lines[2].strip()
    return fallacious, category, specific_type

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

    indices = []
    statements = []
    pred_detection = []
    pred_categories = []
    pred_classes = []
    gt_detection = []
    gt_categories = []
    gt_classes = []
    counter = 0
    
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

    # Create dataset and dataloader
    dataset = FallacyDataset(data, mode=MODE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # For collecting results
    all_results = []

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
            fallacious, category, specific_type = extract_answers(output)
            all_results.append({
                'index': batch['index'][i],
                'statement': batch['statement'][i],
                'pred_detection': fallacious,
                'pred_categories': category,
                'pred_classes': specific_type,
                'gt_detection': batch['gt_detection'][i],
                'gt_categories': batch['gt_category'][i],
                'gt_classes': batch['gt_class'][i]
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join('results', f"{MODE}_{args.model}.csv"), index=False)
    print(f"Results saved to {os.path.join('results', f'{MODE}_{args.model}.csv')}")
    