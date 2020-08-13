import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
'''
*   Plot the Accuracy of Centralized HDC vs FL HDC
'''
# Load each csv file by pd.read_csv

# retrain + FL with K=100
retrain_FL_df = pd.read_csv(os.path.join(
    os.path.dirname(__file__), 're_Avg_accuracy.csv'))
retrain_FL = np.array(retrain_FL_df.iloc[3, 1:])

# FL with K=100
FL_df = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'FLHDC', 'Global_model', 'Avg_accuracy.csv'))
FL = np.array(FL_df.iloc[3, 1:])
# centralized HDC
centralized_df = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Centralized HDC', 'Result', 'train_8000+4000/Accuracy.csv'))
centralized = np.array(centralized_df.iloc[0, 1:])
# retrain + centralized
re_centralized_df = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Centralized HDC', 'Result', 'Retrain_8000+4000/Accuracy.csv'))
re_centralized = np.array(re_centralized_df.iloc[0, 1:])
# Plot the accuracy of 4 cases
dim = [1000, 2000, 5000, 10000]
plt.plot(dim[:], centralized[:], color='b',
         marker='o', linestyle='--', label='One-shot')
plt.plot(dim[:], FL[:], color='b',
         marker='o', linestyle='-', label='FL HDC')
plt.plot(dim[:], re_centralized[:], color='r',
         marker='o', linestyle='--', label='Retrain')
plt.plot(dim[:], retrain_FL[:], color='r',
         marker='o', linestyle='-', label='Retrain + FL HDC')
# plt.axhline(y=0.926,  color='k', linestyle='--',label='FedELMS')

plt.legend()
plt.xticks(dim[:])
plt.grid()
plt.xlabel('Dimension')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(os.path.dirname(__file__), "cenvsFL.png"))
