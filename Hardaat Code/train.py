import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import ChaoticNet
from utils import k_fold_split, optimize_chaotic_params

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def run_training(data, labels, n_epochs=100, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimize chaotic parameters
    optimal_params = optimize_chaotic_params(data)
    print(f"Optimal chaotic parameters: {optimal_params}")
    
    results = []
    for fold, (train_data, train_labels, val_data, val_labels) in enumerate(k_fold_split(data, labels)):
        print(f"Fold {fold+1}")
        
        # Convert to PyTorch tensors
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_data = torch.tensor(val_data, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=batch_size)
        
        # Initialize model
        input_size = train_data.shape[1] * 4  # Multiply by 4 due to feature extraction
        hidden_size = 64
        output_size = len(torch.unique(train_labels))
        model = ChaoticNet(input_size, hidden_size, output_size).to(device)
        
        # Set initial chaotic parameters
        with torch.no_grad():
            model.initial_cond.copy_(torch.tensor(optimal_params[0]))
            model.epsilon.copy_(torch.tensor(optimal_params[1]))
            model.threshold.copy_(torch.tensor(optimal_params[2]))
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(n_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")
        results.append(accuracy)
    
    print(f"Average Validation Accuracy: {sum(results)/len(results):.4f}")