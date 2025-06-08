import json
import random
import logging
from retry import retry
from transformers import pipeline, AutoConfig

from tenacity import retry, stop_after_attempt, wait_fixed
import sys
import pandas as pd
import os

random.seed(4)

def load_statements(filename: str) -> list:
    with open(filename, "r") as f:
        statements = json.load(f)
    print(f"Loaded {len(statements)} statements from {filename}")
    return statements

def prompt_zeroshot(text:str) -> str:
    messages = [
        {"role": "system", "content": "Your task is to simply and promptly give bare answer the next 3 questions. The answer needs to be in the following format, each on a new line: \n1. <Yes/No>\n2. <Fallacy Category>\n3. <Specific Type>.\n Do not generate anything beyond these three lines. Do not explain or continue after the third line."},
        {"role": "user", "content": text},
        {"role": "user", "content": "Is the text fallacious? Only answer with 'yes' or 'no'."},
        {"role": "user", "content": "What category of fallacy is it? You only have to answer with one of the following labels: 'Ad Hominem', 'Appeal to Authority', 'False cause', 'Slogan', 'Appeal to Emotion' or 'None'."},
        {"role": "user", "content": "What specific kind of fallacy is it? You only have to answer with one of the following labels or other labels you see fit: 'General', 'False Authority without evidence', 'Slippery slope', 'None' etc."}
    ]

    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt += "\nAssistant:\n"  # hint model it's time to respond

    return prompt

def prompt_oneshot(text:str) -> str:
    example = (
        "The American people see this debt, and they know it's got to come down.\n"
        "1. Yes\n"
        "2. Fallacy of Credibility\n"
        "3. Appeal to Authority\n"
    )

    messages = [
        {   "role": "system",
            "content": "Your task is to simply and promptly give bare answer the next 3 questions. The answer needs to be in the following format, each on a new line: \n1. <Yes/No>\n2. <Fallacy Category>\n3. <Specific Type>.\n Do not generate anything beyond these three lines. Do not explain or continue after the third line."
        },
        {"role": "user", "content": "Below is one example of how to answer the task:"},
        {"role": "user", "content": example},
        {"role": "user", "content": "Now answer the following statement:"},
        {"role": "user", "content": text},
        {"role": "user", "content": "Is the text fallacious? Only answer with 'yes' or 'no'."},
        {"role": "user", "content": "What category of fallacy is it? You only have to answer with one of the following labels: 'Ad Hominem', 'Appeal to Authority', 'False cause', 'Slogan', 'Appeal to Emotion' or 'None'."},
        {"role": "user", "content": "What specific kind of fallacy is it? You only have to answer with one of the following labels or other labels you see fit: 'General', 'False Authority without evidence', 'Slippery slope', 'None' etc."}
    ]

    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt += "\nAssistant:\n"  # hint model it's time to respond

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

def process_statements(statements: list, generator, mode) -> list:
    results = []
    for text in statements:
        if mode == "zero-shot":
            prompt = prompt_zeroshot(text)
        elif mode == "one-shot":
            prompt = prompt_oneshot(text)
        output = generator(prompt, max_new_tokens=128)[0]["generated_text"]
        print(output)
        fallacious, category, specific_type = extract_answers(output)
        results.append({
            "input_statement": text,
            "fallacious": fallacious,
            "category": category,
            "specific_type": specific_type,
            "mode": mode,
        })
    return results

def save_results(results: list, filename: str):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} results to {filename}")

def zeroshot(generator, model):
    mode = "zero-shot"
    statements = load_statements("statements.json")
    results = process_statements(statements, generator, mode)
    save_results(results, f"zeroshot_answers_final_{model}.json")

def oneshot(generator,model):
    mode = "one-shot"
    statements = load_statements("statements.json")
    results = process_statements(statements, generator, mode)
    save_results(results, f"one_answers_final_{model}.json")



if __name__ == "__main__":
    MODE = "zero-shot"  # or "one-shot"

    #GENERATiION CONFIGURATION
    generation_models = {
    # "phi-4": "microsoft/phi-4",#apparently is a text generation model and does not support question-answering tasks  
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


    for model in generation_models:
        selected_model = model
        model_name = generation_models[selected_model]
        print(f"Using model: {model_name}")

        try:
            config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B")
            print("Access successful!")
        except Exception as e:
            print(f"Still having issues: {e}")


        generator = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

        # loop through the data
        for index, row in data.iterrows():
            indices.append(index)
            statements.append(row['snippet'])
            gt_detection.append(row['fallacy_detection'])
            gt_categories.append(row['category'])
            gt_classes.append(row['class'])

            snippet = row['snippet']
            detection = row['fallacy_detection']
            category = row['category']
            specific_type = row['class']
            print(f"Processing snippet: {snippet}")

            answer = process_statements([snippet], generator, MODE)
            print(answer)

            pred_detection.append(answer[0]['fallacious'])
            pred_categories.append(answer[0]['category'])
            pred_classes.append(answer[0]['specific_type'])

            break
        
        break

    # Save the results to a CSV file
    results_df = pd.DataFrame({
        'index': indices,
        'statement': statements,
        'pred_detection': pred_detection,
        'pred_categories': pred_categories,
        'pred_classes': pred_classes,
        'gt_detection': gt_detection,
        'gt_categories': gt_categories,
        'gt_classes': gt_classes
    })	    
    print(results_df)
    results_df.to_csv(os.path.join(f"results_{selected_model}.csv"), index=False)