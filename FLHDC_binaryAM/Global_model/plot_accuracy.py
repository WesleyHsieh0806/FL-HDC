import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
'''
* Plot the accuracy of each parameter setup
'''
# FL-Binary AM
Dir = os.path.dirname(__file__)
# total_C_result contains two numpy arrays, total_C_result[0]: the accuracy of C=5*e-3 * nj * # of feature/#of class
# total_C_result[1]: the accuracy of C=7*e-3 * nj * # of feature/#of class
total_C_result = []
for K in [20]:
    for dim in [10000]:
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                total_C_result.append(np.array(line.strip().strip(
                    ',').split(','), dtype=float))
# FL SecureHD
Dir2 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'FLHDC_IntegerAM', 'Global_model')
for K in [100]:
    for dim in [10000]:
        with open(os.path.join(Dir2, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list

                accuracy = np.array(line.strip().strip(
                    ',').split(','), dtype=float)
# Plot the accuracy of FLHDC_Binary
x = [i for i in range(len(total_C_result[1]))]
y = [0.01*i for i in range(78, 90)]
plt.plot(x, total_C_result[1], color='r', marker='o', linewidth=3,
         label="FLHDC-Binary")

# plot the accuracy of FLHDC-Integer Framework
plt.plot(x, accuracy[:len(x)], color='yellowgreen', linewidth=3,
         linestyle='-', label='FLHDC-SecureHD')
plt.legend()
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.xlabel('Retrain iteration')
plt.ylabel('Accuracy')
plt.title("Integer vs Binary")
plt.savefig(os.path.join(Dir, "dim10000_K20.png"))
