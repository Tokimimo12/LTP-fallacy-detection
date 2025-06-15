# LTP-fallacy-detection
Welcome to the repository investigating the research question: are top-down fallacy detection and classification are more effective than bottom-up methods?

Before running any code please set up a Python `venv` and install the dependencies in `requirements.txt`.

# Prompting
To collect the prompting results, please `cd prompting` and then run one of the folloiwng files:
1. hierarchical_new.py (for the Hierarchical prompting setup)
2. new_prompting_fixed.py (for the All-At-Once prompting setup)
3. flattened_prompting.py (for the Flattened prompting setup)

Once results have been collected please run the following to collect evaluation metrics:
1. evaluate_answers_hierarchical.py (for ther Hierarchical prompting)
2. evaluate_answers_simple.py (for the All-At-Once prompting)
3. evaluate_answers_flatten.py (for the Flattened prompting.)
Results will be saved as json files in the results folder under prompting.

We experiment using Phi-4, Llama, TinyLlama, Llama-instruct, MistralAI, and Menda. Note that an access key is required to run the Llama models.

# Model Training
We provide the best model for fallacy detection and fallacy classification in this Google Drive: "https://drive.google.com/drive/folders/17F8gSh7aDWhoBj3dAN0laxDwwQLeAXzO?usp=sharing". The models can be downloaded and then run according to the following instructions.
1. Place downloaded models in the `Saved Models` folder
2. Place test data in the folder `data/MM_USED_fallacy/test`. Do note that this data must be a csv with the format ["snippet", "fallacy_detection", "category", "class"], where `snippet` is the text input to classify, `fallacy_detection` is the label for if the text is fallacious (with 1 representing a fallacy), `category` for the category of fallacy label, and `class` for the class of fallacy label.
3. The `eval_trained_models.py` file can then be run, setting the arguments `--model_filename` and `--test_filename` to their respective names.
4. The results are then printed with further metrics saved under `Saved Metrics`