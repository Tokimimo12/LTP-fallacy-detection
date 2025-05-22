import os
from openai import OpenAI
import random
from retry import retry
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
import pandas as pd
# from Preprocessing.preprocess_data_reddit import preprocess_dataset
from tenacity import retry, stop_after_attempt, wait_fixed


# opensource AI models
# Deepseek V3
# Phi 4
# https://huggingface.co/MaziyarPanahi/calme-3.2-instruct-78b


with open("statements.json", "r") as f:
    statements = json.load(f)
    print("Loaded", len(statements), "statements from statements.json")

for statement in statements:
    print("Processing statement:", statement)


token = os.environ['MY_OPENAI_TOKEN']
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Hello, what's 2 + 2?"}
        ]
    )
    print("Response test:", response.choices[0].message.content)
except Exception as e:
    print("Connection or API call failed:", e)



# response = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "",
#         },
#         {
#             "role": "user",
#             "content": "What is the capital of France?",
#         }
#     ],
#     temperature=1,
#     top_p=1,
#     model=model
# )

# print(response.choices[0].message.content)
# with open("response.json", "w") as f:
#     json.dump(response.model_dump(), f, indent=4)



random.seed(4)
# @retry()
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_completion_zeroshot(text):
    mode = 'ZERO-SHOT'
    messages = [
        {"role": "system", "content": "Your task is to answer the next 3 questions. Is the following text fallacious? Answer with 'yes' or 'no'."},
        {"role": "user", "content": text},
        {"role": "system", "content": "What category of fallacy is it? You have to answer with one of the following labels: 'Ad Hominem', 'Appeal to Authority', 'False cause', 'Slogan', 'Appeal to Emotion' or 'None'."},
        {"role": "system", "content": "What specific kind of fallacy is it? You have to answer with one of the following labels or other labels you see fit: 'General', 'False Authority without evidence', 'Slippery slope', 'None' etc."}
    ]

    for msg in messages:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    
    print ("Do I get here??? after printing messages")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
    except Exception as e:
        print("Error during model call:", e)
        raise
    print ("Do I get here??? after response")

    answer_text = response.choices[0].message.content.strip()
    answers = [a.strip() for a in answer_text.split('\n') if a.strip()]
    print("Response:", answer_text)
    print("Answers:", answers)

    # Defensive check if fewer than 3 lines are returned
    while len(answers) < 3:
        answers.append("Unknown")

    data = {
        "input_statement": text,
        "fallacious": answers[0],
        "category": answers[1],
        "specific_type": answers[2],
        "mode": mode,
    }

    # # Save to JSON file
    # with open("test.json", 'w') as f:
    #     json.dump(data, f, indent=4)
    print ("Do I get here??? after datasss")
    return data

    


all_results = []

# with open("statements.json", "r") as f:
#     statements = json.load(f)
#     print("Loaded", len(statements), "statements from statements.json")

for statement in statements:
    print("Processing statement:", statement)
    result = create_completion_zeroshot(statement)
    all_results.append(result)

# Save all results to a single JSON file
with open("test.json", 'w') as f:
    json.dump(all_results, f, indent=4)

print("Saved", len(all_results), "results to test.json")




# def create_completion_oneshot(questions):
#     mode = 'ONE-SHOT'

#     messages = [
#         {"role": "system", "content": "Your task is to answer the following 2 questions using one word."},
#         {"role": "user", "content": questions[0][0]}, 
#         {"role": "assistant", "content": "Paris"},  
#         {"role": "user", "content": questions[0][1]}, 
#         {"role": "assistant", "content": "Eiffel Tower"} 
#     ]

#     print("Messages:")
#     for msg in messages:
#         print(f"{msg['role'].capitalize()}: {msg['content']}")

#     # Make the API call
#     completion = client.chat.completions.create(
#         model=model,
#         messages=messages
#     )

#     return completion, mode




# def zero_shot():
#     with open("questions.json", "r") as f:
#         data = json.load(f)

#     results = []

#     for item in data["zero-shot"]:

#         text_to_check = item[0]  
#         completion, mode = create_completion_zeroshot(text_to_check)
#         final_answer = completion.choices[0].message.content.strip()

#         print("Input:", text_to_check)
#         print("Response:", final_answer)

#         results.append({
#             "mode": mode,
#             "question": text_to_check,
#             "answer": final_answer
#         })

#     with open("zero-shot.json", "w") as f:
#         json.dump(results, f, indent=4)


# def one_shot():

#     train_data = pd.read_csv("data/RedditDataset/train_pre_processed.csv")
#     val_data = pd.read_csv("data/RedditDataset/dev_pre_processed.csv")
#     test_data = pd.read_csv("data/RedditDataset/test_pre_processed.csv")    


#     print(test_data[["overall_fallacy_flag", "overall_fallacy_type"]])





#     # with open("questions.json", "r") as f:
#     #     data = json.load(f)

#     # print("Keys found:", data.keys())  # Should include 'one-shot'

#     # # Extract the one-shot data
#     # one_shot_questions = data["one-shot"]  # <- Use exactly this spelling and hyphen

#     # # Check if it's loading correctly
#     # for pair in one_shot_questions:
#     #     print("Q1:", pair[0])
#     #     print("Q2:", pair[1])
#     # Dictionary to store question-answer pairs

#     # results = {}

#     # # Iterate over each question pair
#     # for question_pair in data["one-shot"]:
#     #     completion, mode = create_completion_oneshot(question_pair)

#     #     # Get the full response from the model
#     #     full_response = completion.choices[0].message.content.strip()

#     #     # Print it to see the output format
#     #     print("Full Response:\n", full_response)

#     #     # Try to split the response into two answers (by newline or punctuation)
#     #     answers = full_response.split('\n')  # Or use .split('.') if separated by periods

#     #     # Basic fallback to handle short outputs
#     #     answer1 = answers[0].strip() if len(answers) > 0 else ""
#     #     answer2 = answers[1].strip() if len(answers) > 1 else ""

#     #     # Save under each question
#     #     results[question_pair[0]] = {"answer": answer1}
#     #     results[question_pair[1]] = {"answer": answer2}

#     #     # Save the answers in a JSON file
#     #     with open("one_shot.json", "w") as f:
#     #         json.dump(results, f, indent=4)






# if __name__ == "__main__":
#     # zero_shot()
#     one_shot()
   




# text_to_check = "What is the capital of the Netherlands?"
# completion, mode = create_completion_fallacy_detection(text_to_check)
# final_answer = completion.choices[0].message.content

# # print("Mode:", mode)
# print("Response:", completion.choices[0].message.content)

# with open("response.json", "w") as f:
#     json.dump(completion.model_dump(), f, indent=4)

# with open("response.txt", "w") as f:
#     f.write("Mode: " + mode + "\n")
#     f.write("Question: " + text_to_check + "\n")
#     f.write("Answer: " + final_answer + "\n")



# if __name__ == "__main__":
#     CALLS = 0
#     mode = 'None'


#     # Evaluation vectors
#     ground_truth = []
#     gpt_preds = []

#     with open('data/sample.json') as filehandle:
#         json_data = json.load(filehandle)

#     TOTAL_CALLS = 1
#     # = len(json_data['test'])
#     for sample in json_data['test']:
#         print('sample:', sample)

        # Text Snippet
        # t1 = 'Text Snippet: '+sample[0]

        # # Label
        # if 'Slipperyslope' == sample[1]:
        #     ground_truth.append(1)
        # elif 'AppealtoMajority' == sample[1]:
        #     ground_truth.append(2)
        # elif 'AppealtoAuthority' == sample[1]:
        #     ground_truth.append(3)
        # elif 'AdHominem' == sample[1]:
        #     ground_truth.append(4)
        # else:
        #     ground_truth.append(0)

        # # Create a completion and a prediction with GPT-Chat
        # completion, mode = create_completion_fallacy_detection(t1)
        # pred = completion.choices[0].message.content

#         if 'slippery' in pred.lower():
#             gpt_preds.append(1)
#         elif 'majority' in pred.lower():
#             gpt_preds.append(2)
#         elif 'authority' in pred.lower():
#             gpt_preds.append(3)
#         elif 'hominem' in pred.lower():
#             gpt_preds.append(4)
#         else:
#             gpt_preds.append(0)
#         '''
#         if 'fallacy' in pred.lower():
#             gpt_preds.append(1)
#         else:
#             gpt_preds.append(0)
# #         '''
#         CALLS += 1
#         print(CALLS,'/',TOTAL_CALLS)
#         print(pred)
#         #print(gpt_preds[-1])

#     mf1 = precision_recall_fscore_support(ground_truth, gpt_preds, average='macro')

#     print(('******************************'))
#     print(('******************************'))
#     print(mode)
#     print(('******************************'))
#     print(('******************************'))
#     print('Macro F1 score in TEST:', mf1)
#     print(('******************************'))
#     print('Confusion matrix')
#     print(confusion_matrix(ground_truth, gpt_preds))
#     print(('******************************'))
#     print(('******************************'))









