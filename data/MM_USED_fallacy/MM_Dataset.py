from torch.utils.data import Dataset

class MM_Dataset(Dataset):
    def __init__(self, snippets, labels):
        self.snippets = snippets
        self.labels = labels

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        snippet = self.snippets[idx]
        detection_label = self.labels[idx][0]
        group_label = self.labels[idx][1]
        classify_label = self.labels[idx][2]
        return snippet, (detection_label, group_label, classify_label)
