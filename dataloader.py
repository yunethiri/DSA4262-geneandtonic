
import torch
from torch.utils.data import Dataset

# Custom Dataset
class MILDataset(Dataset):
    def __init__(self, bags, labels):
        
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        bag = torch.tensor(bag, dtype=torch.float32)

        if self.labels is not None:  # Check if labels exist
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            return bag, label  # Return bag and label if it exists

        return bag