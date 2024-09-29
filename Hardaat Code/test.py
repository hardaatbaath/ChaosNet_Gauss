import torch
from model import ChaoticNet

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Main script to run everything

if __name__ == "__main__":
    # Load your data here
    # data = ...
    # labels = ...
    
    # Run training
    run_training(data, labels)
    
    # For testing, you would typically load your best model and test data
    # model = ChaoticNet(...)
    # model.load_state_dict(torch.load('best_model.pth'))
    # test_loader = ...
    # test(model, test_loader, device)