import argparse
import torch
from torch.optim import Adam
from sklearn.model_selection import KFold
from utils import load_data, save_parameters
from ChaosNet.model import ChaosNetModel

def train_chaosnet(X_train, y_train, num_folds=5, num_epochs=100, learning_rate=0.01):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    best_params = None
    best_score = float('-inf')

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}/{num_folds}")

        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = ChaosNetModel(num_features=X_train.shape[1])
        optimizer = Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        X_train_fold = torch.from_numpy(X_train_fold).float()
        y_train_fold = torch.from_numpy(y_train_fold).long()
        X_val_fold = torch.from_numpy(X_val_fold).float()
        y_val_fold = torch.from_numpy(y_val_fold).long()

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            pred = model(X_train_fold)
            loss = criterion(pred, y_train_fold)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_fold)
            val_loss = criterion(val_pred, y_val_fold)
            val_score = (val_pred.argmax(dim=1) == y_val_fold).float().mean().item()

        print(f"Validation Score: {val_score}")

        if val_score > best_score:
            best_score = val_score
            best_params = model.state_dict()

    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ChaosNet model with k-fold cross-validation.")
    parser.add_argument('--data_path', type=str, default="/Data", help='Path to the training data directory')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs per fold')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')

    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.data_path)

    optimal_params = train_chaosnet(X_train, y_train, num_folds=args.num_folds,
                                    num_epochs=args.num_epochs, learning_rate=args.learning_rate)

    save_parameters(optimal_params, "chaosnet_params.pth")

    print("Training completed. Optimal parameters saved.")