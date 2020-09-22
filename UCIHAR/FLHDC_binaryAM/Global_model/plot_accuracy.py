import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
'''
* Plot the accuracy of each parameter setup
'''
# FL-Binary AM
Dir = os.path.dirname(__file__)
# result: accuracy of FL_BinaryAM
result = []
for K in [20]:
    for dim in [10000]:
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+"_lr.csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                result.append(np.array(line.strip().strip(
                    ',').split(','), dtype=float))
result = np.array(result)
result = np.average(result, axis=0)
# Plot the accuracy of FLHDC_Binary
x = [i for i in range(len(result))]
y = [0.01*i for i in range(78, 90)]
plt.plot(x, result[:len(result)], color='r', marker='o', linewidth=3,
         label="FLHDC-Binary")
plt.legend()
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.xlabel('Retrain Epoch')
plt.ylabel('Accuracy')
plt.title("FLHDC_Binary on UNICHAR")
plt.savefig(os.path.join(Dir, "dim10000_K20.png"))
plt.close()
