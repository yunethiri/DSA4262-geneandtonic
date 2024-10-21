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

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttentionPooling, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, H, masks):
        H = H.permute(1, 0, 2)  # Change to [max_length, batch_size, hidden_dim]
        A, _ = self.multihead_attention(H, H, H, key_padding_mask=(masks == 0))
        A = A.permute(1, 0, 2)  # Change back to [batch_size, max_length, hidden_dim]
        
        # Aggregate the attention outputs (mean pooling)
        M = A.sum(dim=1) / masks.sum(dim=1, keepdim=True)  # Shape: [batch_size, hidden_dim]
        return M

class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        logits = self.fc(x)  # Output raw logits
        return logits

class MILModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate):
        super(MILModel, self).__init__()
        self.instance_encoder = InstanceEncoder(input_dim, hidden_dim)
        self.attention_pooling = MultiHeadAttentionPooling(hidden_dim, num_heads)
        self.classifier = Classifier(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, bags, masks):
        # bags: [batch_size, max_length, input_dim]
        # masks: [batch_size, max_length]
        batch_size, max_length, input_dim = bags.shape
        bags_flat = bags.view(-1, input_dim)
        H = self.instance_encoder(bags_flat)  # [batch_size * max_length, hidden_dim]
        H = H.view(batch_size, max_length, -1)  # [batch_size, max_length, hidden_dim]
        # Apply masks to H
        H = H * masks.unsqueeze(2)

        # Apply dropout before pooling
        H = self.dropout(H)
        M = self.attention_pooling(H, masks)   # [batch_size, hidden_dim]

        # Apply dropout before classification
        M = self.dropout(M)
        outputs = self.classifier(M).squeeze(1)  # [batch_size]

        return outputs