import torch
import torch.nn as nn

class ChaoticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChaoticNet, self).__init__()
        
        # Chaotic parameters
        self.initial_cond = nn.Parameter(torch.tensor(0.5))
        self.epsilon = nn.Parameter(torch.tensor(0.01))
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # Apply chaotic transformation
        trajectory = self.compute_trajectory(self.initial_cond.item(), self.threshold.item(), 1000)
        x = self.extract_features(x, trajectory, self.epsilon.item())
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def compute_trajectory(self, initial_cond, threshold, length):
        trajectory = torch.zeros(length)
        trajectory[0] = initial_cond
        for i in range(1, length):
            x = trajectory[i-1]
            trajectory[i] = torch.where(x < threshold, x / threshold, (1 - x) / (1 - threshold))
        return trajectory
    
    def extract_features(self, x, trajectory, epsilon):
        features = []
        for sample in x:
            sample_features = []
            for value in sample:
                idx = torch.argmin(torch.abs(trajectory - value))
                ttss = torch.mean((trajectory[:idx] > 0.5).float())
                energy = torch.sum(trajectory[:idx]**2)
                tt = idx.float()
                entropy = -torch.sum(trajectory[:idx] * torch.log2(trajectory[:idx] + 1e-10))
                sample_features.extend([ttss, energy, tt, entropy])
            features.append(sample_features)
        return torch.tensor(features, dtype=torch.float32)