from .feature_extractor import transform
from .chaotic_sampler import compute_trajectory

class ChaosNet:
    def __init__(self, equation, initial_params, initial_cond, trajectory_len, epsilon):
        self.equation = equation
        self.params = initial_params
        self.initial_cond = initial_cond
        self.trajectory_len = trajectory_len
        self.epsilon = epsilon
    
    def fit(self, X, y):
        # This method would be implemented to optimize the parameters
        # For now, we'll just use the initial parameters
        pass
    
    def transform(self, X):
        return transform(X, self.initial_cond, self.trajectory_len, self.epsilon, self.params, self.equation)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)