import json
import os

if __name__ == "__main__":
    answers_dir = 'answers'

    for file in os.listdir(answers_dir):
        if file.endswith('.json'):
            file_path = os.path.join(answers_dir, file)
            # parse file name by '_'
            file_name_parts = file.split('_')
            print(file_name_parts)

            with open(file_path, 'r') as f:
                data = json.load(f)
                # Process the data if needed
                print(f"Processed {file}")

            for item in data:
                statement = item.get('input_statement')
                detection = item.get('fallacious')
                category = item.get('category')
                fall_type = item.get('specific_type')
                mode = item.get('mode')
        break

    # correct_count = 0
    # for question, answer in answers.items():
    #     if question in ground_truth and ground_truth[question] == answer:
    #         correct_count += 1

    # total_questions = len(ground_truth)
    # accuracy = correct_count / total_questions if total_questions > 0 else 0

    # print(f"Total Questions: {total_questions}")
    # print(f"Correct Answers: {correct_count}")
    # print(f"Accuracy: {accuracy:.2%}")