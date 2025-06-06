import json
import random
import logging
from retry import retry
from transformers import pipeline, AutoConfig

from tenacity import retry, stop_after_attempt, wait_fixed
import sys



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
        "And there are other ways of squeezing this budget without constantly picking on our senior citizens and the most vulnerable in American life.\n"
        "1. Yes\n"
        "2. Appeal to Authority\n"
        "3. False Authority without evidence"
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


    for model in generation_models:
        selected_model = model
        model_name = generation_models[selected_model]


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
        zeroshot(generator, selected_model)
        oneshot(generator, selected_model)





   # selected_model_key = sys.argv[1] if len(sys.argv) > 1 else "llama"

    # hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # if selected_model_key not in generation_models:
    #     raise ValueError(f"Model '{selected_model_key}' not supported. Choose from: {list(generation_models.keys())}")

    # model_name = generation_models[selected_model_key]




# model_name = "microsoft/phi-4"  
# generator = pipeline(
#     "text-generation",
#     model=model_name,
#     model_kwargs={"torch_dtype": "auto"},
#     device_map="auto",
# )
    # with open(filename, "r") as f:
    #     statements = json.load(f)
    # print(f"Loaded {len(statements)} statements from {filename}")
# results_zeroshot = []

# for text in statements:

#     messages = [
#         {"role": "system", "content": "Your task is to simply and promptly give bare answer the next 3 questions. The answer needs to be in the following format, each on a new line: \n1. <Yes/No>\n2. <Fallacy Category>\n3. <Specific Type>.\n Do not generate anything beyond these three lines. Do not explain or continue after the third line."},
#         {"role": "user", "content": text},
#         {"role": "user", "content": "Is the text fallacious? Only answer with 'yes' or 'no'."},
#         {"role": "user", "content": "What category of fallacy is it? You only have to answer with one of the following labels: 'Ad Hominem', 'Appeal to Authority', 'False cause', 'Slogan', 'Appeal to Emotion' or 'None'."},
#         {"role": "user", "content": "What specific kind of fallacy is it? You only have to answer with one of the following labels or other labels you see fit: 'General', 'False Authority without evidence', 'Slippery slope', 'None' etc."}
#     ]


#     prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])


#     prompt += "\nAssistant:\n"  # hint model it's time to respond



#     outputs = generator(prompt, max_new_tokens=128)

#     print(outputs[0]["generated_text"])

#     answer = outputs[0]["generated_text"]


#     lines = answer.strip().splitlines()

#     try:
#         # Find the index of the assistant's response and save the 3 answers right after it
#         assistant_index = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("assistant:"))
#         next_lines = lines[assistant_index + 1 : assistant_index + 4]
#         if len(next_lines) < 3:
#             raise ValueError("Not enough lines returned by model.")
#         fallacious, category, specific_type = [line.strip() for line in next_lines]
#     except (StopIteration, IndexError, ValueError):
#         # otherwise just save the last 3 lines of your response
#         lines = answer.strip().splitlines()[-3:]  # last 3 lines expected
#         fallacious = lines[0].strip()
#         category = lines[1].strip()
#         specific_type = lines[2].strip()


#     results_zeroshot.append({
#             "input_statement": text,
#             "fallacious": fallacious,
#             "category": category,
#             "specific_type": specific_type,
#             "mode": mode,
#         })

# with open("zeroshot_answers.json", "w") as f:
#     json.dump(results_zeroshot, f, indent=4)





# def convert_messages_to_prompt(messages):
#     """Convert OpenAI-style chat messages to a single string prompt"""
#     prompt = ""
#     for msg in messages:
#         role = msg["role"].capitalize()
#         content = msg["content"]
#         prompt += f"{role}: {content}\n"
#     return prompt

# @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# def create_completion_zeroshot(text):
#     mode = 'ZERO-SHOT'
#     messages = [
#         {"role": "system", "content": "Your task is to simply and promptly give bare answer the next 3 questions in the following format, each on a new line: \n1. <Yes/No>\n2. <Fallacy Category>\n3. <Specific Type>."},
#         {"role": "user", "content": text},
#         {"role": "user", "content": "Is the text fallacious? Only answer with 'yes' or 'no'."},
#         {"role": "user", "content": "What category of fallacy is it? You only have to answer with one of the following labels: 'Ad Hominem', 'Appeal to Authority', 'False cause', 'Slogan', 'Appeal to Emotion' or 'None'."},
#         {"role": "user", "content": "What specific kind of fallacy is it? You only have to answer with one of the following labels or other labels you see fit: 'General', 'False Authority without evidence', 'Slippery slope', 'None' etc."}
#     ]

#     prompt = convert_messages_to_prompt(messages)

    
#     try:
#         # outputs = pipeline(messages, max_new_tokens=128)
#         response = generator(messages, max_new_tokens=200, do_sample=True)
#         print("Last Line Response:", response[0]["generated_text"][-1])

#         print("==============Full response================")
#         print(response[0]["generated_text"])

#         print("===============End responses============================")
        
#     except Exception as e:
#         print("Error during model call:", e)
#         raise

#     generated_text = response[0]["generated_text"]  # Get only the assistant's response text
#     answer_text = generated_text.strip()
#     answers = [a.strip() for a in answer_text.split('\n') if a.strip()]
#     print("answer_text:", answer_text)
#     print("answers:", answers)

#     # while len(answers) < 3:
#     #     answers.append("Unknown")

#     data = {
#         "input_statement": text,
#         "fallacious": answers[0],
#         "category": answers[1],
#         "specific_type": answers[2],
#         "mode": mode,
#     }

#     return data

# def zero_shot():
#     all_results = []
#     for statement in statements:
#         print("Processing statement:", statement)
#         result = create_completion_zeroshot(statement)
#         all_results.append(result)

#     with open("test_zeroshot.json", 'w') as f:
#         json.dump(all_results, f, indent=4)

#     print("Saved", len(all_results), "results to test_zeroshot.json")






# @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# def create_completion_oneshot(text, example):
#     mode = 'ONE-SHOT'
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a helpful assistant that classifies logical fallacies. "
#                 "Below is one example of how to answer the task:"
#                 f"{example}"
#                 "\nNow answer the following statement:"
#             )
#         },
#         {"role": "user", "content": f'Statement: "{text}"\n1. Is this fallacious? Only answer with "yes" or "no"'},
#         {"role": "system", "content": "What category of fallacy is it? You only have to answer with one of the following labels: 'Ad Hominem', 'Appeal to Authority', 'False cause', 'Slogan', 'Appeal to Emotion' or 'None'."},
#         {"role": "system", "content": "What specific kind of fallacy is it? You only have to answer with one of the following labels or other labels you see fit: 'General', 'False Authority without evidence', 'Slippery slope', 'None' etc."}
#     ]

#     prompt = convert_messages_to_prompt(messages)

#     try:
#         response = generator(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
#     except Exception as e:
#         print("Error during model call:", e)
#         raise

#     answer_text = response.strip().replace(prompt.strip(), "").strip()
#     answers = [a.strip() for a in answer_text.split('\n') if a.strip()]
#     print("answer_text:", answer_text)
#     print("answers:", answers)

#     while len(answers) < 3:
#         answers.append("Unknown")

#     data = {
#         "input_statement": text,
#         "fallacious": answers[0],
#         "category": answers[1],
#         "specific_type": answers[2],
#         "mode": mode,
#     }

#     return data

# def one_shot():
#     all_results = []
#     with open("examples.json", "r") as f:
#         examples = json.load(f)
#     example = random.choice(examples)

#     for statement in statements:
#         print("Processing statement:", statement)
#         result = create_completion_oneshot(statement, example)
#         all_results.append(result)

#     with open("test_oneshot.json", 'w') as f:
#         json.dump(all_results, f, indent=4)

#     print("Saved", len(all_results), "results to test_oneshot.json")




# if __name__ == "__main__":
#     with open("statements.json", "r") as f:
#         statements = json.load(f)
#         print("Loaded %d statements from statements.json", len(statements))

#     zero_shot()
#     # one_shot()
