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