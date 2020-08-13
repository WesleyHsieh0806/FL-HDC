import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
'''
* Plot the accuracy of each parameter setup
'''
Dir = os.path.dirname(__file__)
accuracy = []
for K in [20, 40, 80, 100, 120]:
    for dim in [1000, 2000, 5000, 10000]:
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                if len(accuracy) == 0:
                    accuracy.append(np.average(np.array(line.strip().strip(
                        ',').split(','), dtype=float), axis=0))
                else:

                    accuracy.append(np.average(np.array(line.strip().strip(
                        ',').split(','), dtype=float)))
accuracy = np.array(accuracy, dtype=np.float64).reshape(5, 4)
accuracy_df = pd.DataFrame(accuracy, columns=["dim"+str(dim) for dim in [
                           1000, 2000, 5000, 10000]], index=[K for K in [20, 40, 80, 100, 120]])
accuracy_df.to_csv(os.path.join(Dir, 'Avg_accuracy.csv'))

client20 = np.array(accuracy_df.iloc[0, :], dtype=np.float64)
client40 = np.array(accuracy_df.iloc[1, :], dtype=np.float64)
client80 = np.array(accuracy_df.iloc[2, :], dtype=np.float64)
client100 = np.array(accuracy_df.iloc[3, :], dtype=np.float64)
client120 = np.array(accuracy_df.iloc[4, :], dtype=np.float64)
dim = [1000, 2000, 5000, 10000]
plt.plot(dim[2:], client20[2:], color='r',
         marker='o', linestyle='-', label='K=20')
plt.plot(dim[2:], client40[2:], color='y',
         marker='o', linestyle='-', label='K=40')
plt.plot(dim[2:], client80[2:], color='g',
         marker='o', linestyle='-', label='K=80')
plt.plot(dim[2:], client100[2:], color='b',
         marker='o', linestyle='-', label='K=100')
plt.plot(dim[2:], client120[2:], color='c',
         marker='o', linestyle='-', label='K=120')
plt.legend()
plt.xticks(dim[2:])
plt.grid()
plt.xlabel('Dimension')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(Dir, "Avg_accuracy.png"))
