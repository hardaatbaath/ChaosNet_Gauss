
import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data
from Codes import k_cross_validation
import os

DATA_NAME = "concentric_circle"
traindata, trainlabel, testdata, testlabel = get_data(DATA_NAME)
FOLD_NO = 5

INITIAL_NEURAL_ACTIVITY = np.arange(0.1, 1.1, 0.1)
D_alpha = np.arange(0.1, 1.1, 0.1)
D_beta = np.arange(0.1, 1.1, 0.1)
EPSILON = np.arange(0.01, 1.01, 0.01)
FSCORE, Q, A, B, EPS, EPSILON = k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, D_alpha, D_beta, EPSILON, DATA_NAME)


PATH = os.getcwd()
RESULT_PATH = PATH + '/SR-PLOTS/' + DATA_NAME + '/NEUROCHAOS-RESULTS/'

# Create the directory if it doesn't exist
os.makedirs(RESULT_PATH, exist_ok=True)

# Create a plot for each alpha and beta combination
for alpha_idx, alpha in enumerate(D_alpha):
    for beta_idx, beta in enumerate(D_beta):
        plt.figure(figsize=(12,8))
        
        # Plot F-scores for all initial neural activities across different epsilon values
        for ina_idx, ina in enumerate(INITIAL_NEURAL_ACTIVITY):
            plt.plot(EPSILON, FSCORE[alpha_idx, beta_idx, ina_idx, :], 
                     label=f'INA = {ina:.2f}', 
                     marker='*', 
                     markersize=5)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('Noise intensity (ε)', fontsize=18)
        plt.ylabel('Average F1-score', fontsize=18)
        plt.title(f'F1-score vs Noise Intensity\nα = {alpha:.2f}, β = {beta:.2f}', fontsize=20)
        plt.ylim(0, 1)
        plt.legend(title='Initial Neural Activity', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot with a unique filename including alpha and beta values
        filename = f"/Chaosnet-{DATA_NAME}-SR_plot_alpha_{alpha:.2f}_beta_{beta:.2f}_INA_comparison.jpg"
        plt.savefig(RESULT_PATH + filename, format='jpg', dpi=200, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

print(f"Plots saved in {RESULT_PATH}")