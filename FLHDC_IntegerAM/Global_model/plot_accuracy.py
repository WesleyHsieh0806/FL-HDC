import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
'''
* Plot the accuracy of each parameter setup
'''
Dir = os.path.dirname(__file__)
for K in [100]:
    for dim in [10000]:
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list

                accuracy = np.array(line.strip().strip(
                    ',').split(','), dtype=float)

# FL with K=100
FL_df = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'FLHDC', 'Global_model', 'Avg_accuracy.csv'))
FL = np.array(FL_df.iloc[3, 1:])
# plot the accuracy between FLHDC vs FLHDC SecureHD(with retrain)
x = [i for i in range(len(accuracy))]
plt.plot(x, accuracy, color='yellowgreen', linewidth=4,
         linestyle='-', label='FLHDC-SecureHD')
plt.hlines(FL[3], x[0], x[-1], colors='b',
           linestyles='--', label='FLHDC-One shot')
plt.xlabel("Retrain iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.title("FLHDC-SecureHD")
plt.savefig(os.path.join(Dir, 'dim10000_K100.png'))
plt.close()
