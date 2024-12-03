import numpy as np
import matplotlib.pyplot as plt
from load_data import get_data
from Codes import k_cross_validation
import os

# Dataset loading
DATA_NAME = "single_variable_classification"
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)

# Importing necessary libraries for classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate a classification model
    
    Parameters:
    - model: Sklearn or XGBoost classifier
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels
    - model_name: Name of the model for printing
    
    Returns:
    - Detailed performance metrics
    """
    # Ensure y is 1D array
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print results
    print(f"\n{model_name} Results:")
    print("Cross-validation F1 Scores:", cv_scores)
    print(f"Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model_name,
        'cv_scores': cv_scores.tolist(),  # Convert to list for JSON serialization
        'test_f1': f1,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

# Initialize models with some hyperparameters
models = [
    ('SVM (RBF Kernel)', SVC(kernel='rbf', random_state=42, probability=True)),
    ('XGBoost', XGBClassifier(eval_metric='logloss', random_state=42)),
    ('Multi-layer Perceptron', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))
]

# Store results for comparison
results = []

# Train and evaluate each model
for name, model in models:
    result = train_and_evaluate_model(model, traindata, trainlabel, testdata, testlabel, name)
    results.append(result)

# Visualize F1 Scores
plt.figure(figsize=(10, 6))
model_names = [r['model'] for r in results]
test_f1_scores = [r['test_f1'] for r in results]

plt.bar(model_names, test_f1_scores)
plt.title(f'Model Comparison - F1 Scores ({DATA_NAME} Dataset)')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Save plot with dataset name
plot_filename = f'results/{DATA_NAME}_model_comparison.png'
plt.savefig(plot_filename)
plt.close()

# Save results to a JSON file
import json
results_filename = f'results/{DATA_NAME}_model_comparison_results.json'
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nPlot saved to: {plot_filename}")
print(f"Results saved to: {results_filename}")