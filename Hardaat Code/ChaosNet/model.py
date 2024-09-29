import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor

class ChaosNetModel(nn.Module):
    def __init__(self, num_features):
        super(ChaosNetModel, self).__init__()
        self.num_features = num_features
        self.initial_cond = nn.Parameter(torch.rand(1))
        self.threshold = nn.Parameter(torch.rand(1))
        self.epsilon = nn.Parameter(torch.rand(1))
        self.trajectory_len = 10000
        self.classifier = nn.Linear(num_features * 4, 2)  # Binary classification

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x):
        features = FeatureExtractor.transform(
            x.numpy(),
            self.initial_cond.item(),
            self.trajectory_len,
            self.epsilon.item(),
            self.threshold.item()
        )
        return torch.from_numpy(features).float()

    def classify(self, features):
        return torch.argmax(self.classifier(features), dim=1)

    def compute_loss(self, x, y):
        outputs = self(x)
        return nn.functional.cross_entropy(outputs, y)