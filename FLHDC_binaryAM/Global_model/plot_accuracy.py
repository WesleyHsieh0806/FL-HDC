import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
'''
* Plot the accuracy of each parameter setup
'''
Dir = os.path.dirname(__file__)
# total_C_result contains two numpy arrays, total_C_result[0]: the accuracy of C=0.2*nj//L
# total_C_result[1]: the accuracy of C=1*nj//L
total_C_result = []
for K in [100]:
    for dim in [10000]:
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                total_C_result.append(np.array(line.strip().strip(
                    ',').split(','), dtype=float))
x = [i for i in range(len(total_C_result[0]))]
y = [0.1*i for i in range(1, 10)]
plt.plot(x, total_C_result[0], color='tab:red',marker='o',label="C = 0.2*nj /# of class")
plt.plot(x, total_C_result[1], color='r', marker='o',label="C = nj /# of class")
plt.legend()
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.xlabel('Retrain iteration')
plt.ylabel('Accuracy')
plt.title("FLHDC-Binary Framework")
plt.savefig(os.path.join(Dir, "dim10000_K100.png"))
