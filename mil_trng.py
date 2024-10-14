import numpy as np
import json  
import pandas as pd 
from sklearn.model_selection import train_test_split  
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, confusion_matrix

# Load your labels
df_labels = pd.read_csv('data.info.txt', delimiter=',')
labels = df_labels['label'].tolist()
labels = list(df_labels['label'])
gene_id = list(df_labels['gene_id'])

# Load your data
data_list = []
with open('dataset0.json', 'r') as file:
    for line in file:
        try:
            data = json.loads(line) 
            data_list.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line}")
            print(f"Error message: {e}")
            

# Build bags
bags = [] 
for dat in (data_list):
    for _, pos in dat.items():  
        for _, seq in pos.items():  
            for _, measurements in seq.items(): 
                bag = []
                for read in measurements:
                    instance = np.array(read).reshape(9,) 
                    bag.append(instance)
                if len(bag) == 0:
                    print(bag) 
                bags.append(bag)
df = pd.DataFrame({
    'gene_id': gene_id,
    'bags': bags,
    'label': labels
})

def sample_with_ratio(group):
    label_1 = group[group['label'] == 1]
    label_0 = group[group['label'] == 0]

    if len(label_0) == 0:
        sample_label_1 = label_1.sample(n=min(len(label_1), 5), random_state=42)
        sample_label_0 = pd.DataFrame()  # No label_0 data
    else:
        max_label_1 = 5 * len(label_0)
        sample_label_1 = label_1.sample(n=min(len(label_1), max_label_1), random_state=42)
        sample_label_0 = label_0.sample(n=min(len(label_0), 2), random_state=42)

    combined_sample = pd.concat([sample_label_1, sample_label_0])
    if len(combined_sample) > 10:
        combined_sample = combined_sample.sample(n=10, random_state=42)
    
    sampled_indices = combined_sample.index
    remaining_data = group.drop(sampled_indices)
    return combined_sample, remaining_data


# grouped = df.groupby('gene_id')
# sampled_data = pd.DataFrame()
# remaining_data = pd.DataFrame()

# for name, group in grouped:
#     sampled_group, remaining_group = sample_with_ratio(group)
#     sampled_data = pd.concat([sampled_data, sampled_group])
#     remaining_data = pd.concat([remaining_data, remaining_group])

# sampled_data.reset_index(drop=True, inplace=True)
# remaining_data.reset_index(drop=True, inplace=True)

# extra_bags = remaining_data['bags'].tolist()
# extra_labels = remaining_data['label'].tolist()

# sampled_bags = sampled_data['bags'].tolist()
# sampled_labels = sampled_data['label'].tolist()
# bags, labels = sampled_bags, sampled_labels

print("data loaded")
print(len(bags))
print(len(labels))
print(len(df))
labels_series = pd.Series(labels)
print(labels_series.value_counts())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Synthetic Data Generation
# bags = []
# labels = []
# num_bags = 100
# for _ in range(num_bags):
#     num_instances = np.random.randint(1, 10)
#     bag = np.random.rand(num_instances, 9)
#     label = np.random.randint(0, 2)
#     bags.append(bag)
#     labels.append(label)
# labels = np.array(labels)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Custom Dataset
class MILDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]
        bag = torch.tensor(bag, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return bag, label

# Custom Collate Function with Padding and Masking
from torch.nn.utils.rnn import pad_sequence

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

# Data Split
bags_train, bags_temp, labels_train, labels_temp = train_test_split(
    bags, labels, test_size=0.4, random_state=42)
bags_val, bags_test, labels_val, labels_test = train_test_split(
    bags_temp, labels_temp, test_size=0.5, random_state=42)

bags_train_pos = []
labels_train_pos = []
bags_train_neg = []
labels_train_neg = []

for bag, label in zip(bags_train, labels_train):
    if label == 1:
        bags_train_pos.append(bag)
        labels_train_pos.append(label)
    else:
        bags_train_neg.append(bag)
        labels_train_neg.append(label)

num_pos = len(labels_train_pos)
num_neg = len(labels_train_neg)
print(f"Number of positive samples in training data: {num_pos}")
print(f"Number of negative samples in training data: {num_neg}")

desired_num_pos = 15000 
desired_num_neg = 40000

from sklearn.utils import resample

# Oversample positive samples
bags_train_pos_resampled, labels_train_pos_resampled = resample(
    bags_train_pos,
    labels_train_pos,
    replace=True,  # Sample with replacement
    n_samples=desired_num_pos,
    random_state=42
)
# Undersample neg examples
bags_train_neg_resampled, labels_train_neg_resampled = resample(
    bags_train_neg,
    labels_train_neg,
    replace=False,  # Sample without replacement
    n_samples=desired_num_neg,
    random_state=42
)

bags_train_resampled = bags_train_pos_resampled + bags_train_neg_resampled
labels_train_resampled = labels_train_pos_resampled + labels_train_neg_resampled

from sklearn.utils import shuffle
bags_train_resampled, labels_train_resampled = shuffle(
    bags_train_resampled, labels_train_resampled, random_state=42
)

bags_train = bags_train_resampled
labels_train = labels_train_resampled

# DataLoaders with multiple workers
batch_size = 32
train_dataset = MILDataset(bags_train, labels_train)
val_dataset = MILDataset(bags_val, labels_val)
test_dataset = MILDataset(bags_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=mil_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=mil_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=mil_collate_fn)

# Model Components
class InstanceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InstanceEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, H, masks):
        A = self.attention(H).squeeze(2)  # Shape: [batch_size, max_length]
        A = A.masked_fill(masks == 0, float('-inf'))
        A = torch.softmax(A, dim=1)
        A = A.unsqueeze(2)  # Shape: [batch_size, max_length, 1]
        M = torch.sum(A * H, dim=1)  # Shape: [batch_size, hidden_dim]
        return M

class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc(x)  # Output raw logits

class MILModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MILModel, self).__init__()
        self.instance_encoder = InstanceEncoder(input_dim, hidden_dim)
        self.attention_pooling = AttentionPooling(hidden_dim)
        self.classifier = Classifier(hidden_dim)

    def forward(self, bags, masks):
        # bags: [batch_size, max_length, input_dim]
        # masks: [batch_size, max_length]
        batch_size, max_length, input_dim = bags.shape
        bags_flat = bags.view(-1, input_dim)
        H = self.instance_encoder(bags_flat)  # [batch_size * max_length, hidden_dim]
        H = H.view(batch_size, max_length, -1)  # [batch_size, max_length, hidden_dim]
        # Apply masks to H
        H = H * masks.unsqueeze(2)
        M = self.attention_pooling(H, masks)   # [batch_size, hidden_dim]
        outputs = self.classifier(M).squeeze(1)  # [batch_size]
        return outputs

# Initialize Model, Loss, Optimizer
input_dim = 9
hidden_dim = 128 #og was 64
learning_rate = 0.001
num_epochs = 50
threshold = 0.5

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
labels_train_array = np.array(labels_train)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train_array), y=labels_train_array)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
pos_weight = class_weights[1] / class_weights[0]

# Define loss function and move model to device
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2)
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
model = MILModel(input_dim, hidden_dim).to(device)

import torch.optim as optim

# Define the SGD optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Added momentum for better convergence


# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for bags_batch, labels_batch, masks_batch in train_loader:
        bags_batch = bags_batch.to(device)
        labels_batch = labels_batch.to(device)
        masks_batch = masks_batch.to(device)
        optimizer.zero_grad()
        outputs = model(bags_batch, masks_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_outputs = []
        val_losses = []
        for bags_batch, labels_batch, masks_batch in val_loader:
            bags_batch = bags_batch.to(device)
            labels_batch = labels_batch.to(device)
            masks_batch = masks_batch.to(device)
            outputs = model(bags_batch, masks_batch)
            loss = criterion(outputs, labels_batch)
            val_losses.append(loss.item())
            outputs = torch.sigmoid(outputs)
            all_labels.extend(labels_batch.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
        
        roc_auc = roc_auc_score(all_labels, all_outputs)
        pr_auc = average_precision_score(all_labels, all_outputs)
        acc = accuracy_score(all_labels, (np.array(all_outputs) > threshold).astype(int))
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {np.mean(train_losses):.4f}, "
              f"Val Loss: {np.mean(val_losses):.4f}, "
              f"Val ROC-AUC: {roc_auc:.4f}, "
              f"Val PR-AUC: {pr_auc:.4f}, "
              f"Val Accuracy: {acc:.4f}")

torch.save(model.state_dict(), 'model_weights.pth')

# Test Evaluation
model.eval()
with torch.no_grad():
    all_labels = []
    all_outputs = []
    for bags_batch, labels_batch, masks_batch in test_loader:
        bags_batch = bags_batch.to(device)
        labels_batch = labels_batch.to(device)
        masks_batch = masks_batch.to(device)
        outputs = model(bags_batch, masks_batch)
        outputs = torch.sigmoid(outputs)
        all_labels.extend(labels_batch.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())

    roc_auc = roc_auc_score(all_labels, all_outputs)
    pr_auc = average_precision_score(all_labels, all_outputs)
    acc = accuracy_score(all_labels, (np.array(all_outputs) > threshold).astype(int))
    cm = confusion_matrix(all_labels, (np.array(all_outputs) > threshold).astype(int))
    print(f"\nTest Set Results:\n"
          f"ROC-AUC: {roc_auc:.4f}\n"
          f"PR-AUC: {pr_auc:.4f}\n"
          f"Accuracy: {acc:.4f}\n"
          f"Confusion Matrix:\n{cm}")

# ROC and PR Curves
# ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# PR Curve
precision, recall, thresholds = precision_recall_curve(all_labels, all_outputs)
plt.figure()
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Confusion Matrix Visualization
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()


model.eval()
with torch.no_grad():
    all_labels = []
    all_outputs = []
    for bags_batch, labels_batch in extra_test_loader:
        outputs = model(bags_batch)
        all_labels.extend(labels_batch.numpy())
        all_outputs.extend(outputs.numpy())

    roc_auc = roc_auc_score(all_labels, all_outputs)
    pr_auc = average_precision_score(all_labels, all_outputs)
    acc = accuracy_score(all_labels, (np.array(all_outputs) > threshold).astype(int))
    cm = confusion_matrix(all_labels, (np.array(all_outputs) > threshold).astype(int))
    print(f"\nTest Set Results:\n"
          f"ROC-AUC: {roc_auc:.4f}\n"
          f"PR-AUC: {pr_auc:.4f}\n"
          f"Accuracy: {acc:.4f}\n"
          f"Confusion Matrix:\n{cm}")

# ROC and PR Curves
# ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# PR Curve
precision, recall, thresholds = precision_recall_curve(all_labels, all_outputs)
plt.figure()
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Confusion Matrix Visualization
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()