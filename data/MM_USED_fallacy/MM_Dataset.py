from torch.utils.data import Dataset
from Model.HTC_utils import get_tree_dicts, get_reverse_dicts

class MM_Dataset(Dataset):
    def __init__(self, snippets, labels, mtl=True):
        self.snippets = snippets
        self.labels = labels
        self.mtl = mtl

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        snippet = self.snippets[idx]
        if self.mtl:
            # Multi-task learning: return all three labels
            detection_label = self.labels[idx][0]
            group_label = self.labels[idx][1]
            classify_label = self.labels[idx][2]
        else:
            # Single-task learning: gotta change it to add extra class
            detection_label = self.labels[idx][0]
            group_label = self.labels[idx][1]
            if detection_label == 0: # non-fallacy
            # If it's a non-fallacy, we classify it as "non-fallacy" (6)
                classify_label = 6
            else:
                classify_label = self.labels[idx][2]
        return snippet, (detection_label, group_label, classify_label)


class HTC_MM_Dataset(Dataset):
    def __init__(self, snippets, labels, name_to_node_id):
        self.snippets = snippets
        self.labels = labels
        self.dict = name_to_node_id
        class_dict, category_dict, detection_dict = get_reverse_dicts()
        self.det_dict = detection_dict
        self.cat_dict = category_dict
        self.class_dict = class_dict

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        snippet = self.snippets[idx]
        detection_label = self.labels[idx][0]
        detection_label = self.det_dict[detection_label]
        group_label = self.labels[idx][1]
        group_label = self.cat_dict[group_label]
        classify_label = self.labels[idx][2]
        classify_label = self.dict[classify_label]
        return snippet, (detection_label, group_label, classify_label)