import numpy as np
from ChaosNet import feature_extractor as CFX
from utils import load_data, load_parameters

def test_chaosnet(X_test, y_test, params):
    # Unpack parameters
    equation_params, initial_cond, epsilon = params[:-2], params[-2], params[-1]
    
    # Extract features using ChaosNet
    features = CFX.transform(X_test, initial_cond, 10000, epsilon, equation_params)
    
    # Simple classifier (e.g., nearest centroid)
    centroids = np.array([features[y_test == c].mean(axis=0) for c in np.unique(y_test)])
    y_pred = np.argmin(np.linalg.norm(features[:, None] - centroids, axis=2), axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    return accuracy

if __name__ == "__main__":
    # Load data
    X_test, y_test = load_data("test_data.csv")
    
    # Load parameters
    params = load_parameters("chaosnet_params.npy")
    
    # Test ChaosNet
    accuracy = test_chaosnet(X_test, y_test, params)
    
    print(f"Test accuracy: {accuracy:.4f}")