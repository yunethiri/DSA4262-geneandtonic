import torch
import torch.nn as nn
import torch.nn.functional as F


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
        logits = self.fc(x)  # Output raw logits
        return logits

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
