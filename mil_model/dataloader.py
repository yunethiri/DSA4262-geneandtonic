
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
    

def mil_collate_fn(batch):
    bags = [item[0] for item in batch]  # List of tensors
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    
    # Pad the sequences
    padded_bags = pad_sequence(bags, batch_first=True)  # Shape: [batch_size, max_length, input_dim]
    
    # Create masks
    lengths = torch.tensor([bag.size(0) for bag in bags])
    max_length = padded_bags.size(1)
    masks = torch.arange(max_length).expand(len(bags), max_length) < lengths.unsqueeze(1)
    masks = masks.float()
    
    return padded_bags, labels, masks


def create_dataloaders(bags_train, labels_train, bags_val, labels_val, bags_test, labels_test, batch_size):
    train_dataset = MILDataset(bags_train, labels_train)
    val_dataset = MILDataset(bags_val, labels_val)
    test_dataset = MILDataset(bags_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=mil_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=mil_collate_fn)
    
    return train_loader, val_loader, test_loader