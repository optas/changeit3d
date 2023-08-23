import torch.nn as nn

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, in_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
