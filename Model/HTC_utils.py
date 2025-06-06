from hierarchicalsoftmax import SoftmaxNode

classes = ["appeal to emotion", "appeal to authority", "ad hominem", "false cause", "slippery slope", "slogans", "no fallacy"]

def get_tree():
    root = SoftmaxNode("root")
    is_fallacy = SoftmaxNode("is fallacy", parent=root)
    no_fallacy = SoftmaxNode("no fallacy", parent=root)

    emotion = SoftmaxNode("fallacy of emotion", parent=is_fallacy)
    logic = SoftmaxNode("fallacy of logic", parent=is_fallacy)
    credibility = SoftmaxNode("fallacy of credibility", parent=is_fallacy)

    appeal_emotion = SoftmaxNode("appeal to emotion", parent=emotion)
    slogans = SoftmaxNode("slogans", parent=emotion)

    appeal_authority = SoftmaxNode("appeal to authority", parent=credibility)
    ad_hominem = SoftmaxNode("ad hominem", parent=credibility)

    false_cause = SoftmaxNode("false cause", parent=logic)
    slippery_slope = SoftmaxNode("slippery slope", parent=logic)

    root.set_indexes()
    # root.render(print=True)

    return root

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

def get_tree_dicts(root, data):
    name_to_node_id = {node.name: root.node_to_id[node] for node in root.leaves}
    # print(name_to_node_id)
    index_to_node_id = {
        i: name_to_node_id[name] for i, name in enumerate(classes)
    }

    return name_to_node_id, index_to_node_id

def post_process_predictions(predictions):
    """
    Convert model predictions to class names using the provided mapping.
    """
    _, index_to_node_id = get_tree_dicts(get_tree(), None)
    node_id_to_index = {v: k for k, v in index_to_node_id.items()}

    processed_predictions = []
    for pred in predictions:
        if isinstance(pred, list):
            # If the prediction is a list, convert each element
            processed_pred = [node_id_to_index.get(item, item) for item in pred]
        else:
            # If it's a single prediction, convert it directly
            processed_pred = node_id_to_index.get(pred, pred)
        processed_predictions.append(processed_pred)
    return processed_predictions
