from torch.utils.data import Dataset

class MM_Dataset(Dataset):
    def __init__(self, snippets, labels):
        self.snippets = snippets
        self.labels = labels

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        return self.snippets[idx], self.labels[idx]
