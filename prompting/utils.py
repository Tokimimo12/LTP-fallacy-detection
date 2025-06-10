
def get_index_dicts():
    class_to_name = {0: "appeal to emotion",  # Appeal to Emotion -> Fallacy of Emotion
        1: "appeal to authority",  # Appeal to Authority -> Fallacy of Credibility
        2: "ad hominem",  # Ad Hominem -> Fallacy of Credibility
        3: "false cause",  # False Cause -> Fallacy of Logic
        4: "slippery slope",  # Slippery Slope -> Fallacy of Logic
        5: "slogans",   # Slogans -> Fallacy of Emotion
        -1: "no fallacy"  # No Fallacy -> No Fallacy
    }

    category_to_name = {
        0: "fallacy of emotion",
        1: "fallacy of credibility",
        2: "fallacy of logic"   
    }

    detection_to_name = {
        1: "is fallacy",
        0: "no fallacy"
    }

    return class_to_name, category_to_name, detection_to_name

def get_reverse_dicts():
    class_to_name, category_to_name, detection_to_name = get_index_dicts()
    name_to_class = {v: k for k, v in class_to_name.items()}
    name_to_category = {v: k for k, v in category_to_name.items()}
    name_to_detection = {v: k for k, v in detection_to_name.items()}

    return name_to_class, name_to_category, name_to_detection

def get_possible_outputs():
    detection_labels = ["Not Fallacious", "Fallacious"]
    category_labels = ["Fallacy of Emotion", "Fallacy of Credibility", "Fallacy of Logic"]
    class_labels = ["Appeal to Emotion", "Appeal to Authority", "Ad Hominem", "False Cause", "Slippery Slope", "Slogans"]

    return detection_labels, category_labels, class_labels

def get_class_to_category():
    class_to_category = {
        "Appeal to Emotion": "Fallacy of Emotion",
        "Appeal to Authority": "Fallacy of Credibility",
        "Ad Hominem": "Fallacy of Credibility",
        "False Cause": "Fallacy of Logic",
        "Slippery Slope": "Fallacy of Logic",
        "Slogans": "Fallacy of Emotion",
        "None": "None"
    }
    return class_to_category

def get_possible_classes_per_category():
    possible_classes = {
        "Fallacy of Emotion": ["Appeal to Emotion", "Slogans"],
        "Fallacy of Credibility": ["Appeal to Authority", "Ad Hominem"],
        "Fallacy of Logic": ["False Cause", "Slippery Slope"],
        "None": ["None"]
    }
    return possible_classes
