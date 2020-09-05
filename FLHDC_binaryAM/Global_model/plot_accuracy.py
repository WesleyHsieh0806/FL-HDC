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
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                result = np.array(line.strip().strip(
                    ',').split(','), dtype=float)
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
x = [i for i in range(len(result))]
y = [0.01*i for i in range(78, 90)]
plt.plot(x, result, color='r', marker='o', linewidth=3,
         label="FLHDC-Binary")

# plot the accuracy of FLHDC-Integer Framework
plt.plot(x, accuracy[:len(x)], color='yellowgreen', linewidth=3,
         linestyle='-', label='FLHDC-SecureHD')
plt.legend()
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.xlabel('Retrain Epoch')
plt.ylabel('Accuracy')
plt.title("Integer vs Binary")
plt.savefig(os.path.join(Dir, "dim10000_K20.png"))
plt.close()

# plot the accuracy between centralized_binary and FLHDC_binary
cen_bin = [0.8098, 0.852, 0.8651, 0.8702, 0.8773, 0.8882, 0.8882, 0.892, 0.896]
plt.plot(range(len(result)), result[:len(
    result)], color='r', marker='o', label="FLHDC-Binary")
# plot the accuracy of Centralized HDC with Binary AM
plt.plot(range(len(cen_bin)), cen_bin, color='r', marker='o', linestyle='--',
         label="Centralized-Binary")
plt.legend()
plt.xticks(range(len(result)))
plt.yticks(y)
plt.grid()
plt.xlabel('Retrain Epoch')
plt.ylabel('Accuracy')
plt.title("Centralized vs FL-Binary")
plt.savefig(os.path.join(Dir, "dim10000_K20_cenvsFLbin.png"))
plt.close()
