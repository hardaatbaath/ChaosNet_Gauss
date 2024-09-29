import numpy as np
import torch
from ChaosNet.model import ChaosNetModel
from utils import load_data, load_parameters

def test_chaosnet(X_test, y_test, params):
    model = ChaosNetModel(num_features=X_test.shape[1])
    model.load_state_dict(params)
    model.eval()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1)

    accuracy = (y_pred == y_test).float().mean().item()
    return accuracy

if __name__ == "__main__":
    # Load data
    X_test, y_test = load_data("test_data.csv")

    # Load parameters
    params = load_parameters("chaosnet_params.pth")

    # Test ChaosNet
    accuracy = test_chaosnet(X_test, y_test, params)

    print(f"Test accuracy: {accuracy:.4f}")