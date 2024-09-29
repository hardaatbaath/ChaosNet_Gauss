import numpy as np
from scipy.optimize import minimize
from ChaosNet import feature_extractor as CFX
from utils import load_data, save_parameters

def objective_function(params, X_train, y_train):
    # Unpack parameters
    equation_params, initial_cond, epsilon = params[:-2], params[-2], params[-1]
    
    # Extract features using ChaosNet
    features = CFX.transform(X_train, initial_cond, 10000, epsilon, equation_params)
    
    # Simple classifier (e.g., nearest centroid)
    centroids = np.array([features[y_train == c].mean(axis=0) for c in np.unique(y_train)])
    y_pred = np.argmin(np.linalg.norm(features[:, None] - centroids, axis=2), axis=1)
    
    # Return negative accuracy as we want to maximize accuracy
    return -np.mean(y_pred == y_train)

def train_chaosnet(X_train, y_train):
    # Initialize parameters (assuming 3 equation parameters for now)
    initial_params = np.random.rand(5)  # 3 equation params + initial_cond + epsilon
    
    # Set bounds for parameters
    bounds = [(0, 1)] * len(initial_params)
    
    # Optimize parameters
    result = minimize(objective_function, initial_params, args=(X_train, y_train), 
                      method='L-BFGS-B', bounds=bounds)
    
    return result.x

if __name__ == "__main__":
    # Load data
    X_train, y_train = load_data("train_data.csv")
    
    # Train ChaosNet
    optimal_params = train_chaosnet(X_train, y_train)
    
    # Save parameters
    save_parameters(optimal_params, "chaosnet_params.npy")
    
    print("Training completed. Optimal parameters saved.")