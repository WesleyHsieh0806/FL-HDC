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
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+"_lr_retraininit.csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                result.append(np.array(line.strip().strip(
                    ',').split(','), dtype=float))
result = np.array(result)
result = np.average(result, axis=0)
# FL SecureHD
accuracy = []
Dir2 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'FLHDC_IntegerAM', 'Global_model')
for K in [100]:
    for dim in [10000]:
        with open(os.path.join(Dir2, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list

                accuracy.append(np.array(line.strip().strip(
                    ',').split(','), dtype=float))
accuracy = np.array(accuracy)
accuracy = np.average(accuracy, axis=0)
# Plot the accuracy of FLHDC_Binary vs FLHDC_SecureHD
x = [i for i in range(len(result))]
y = [0.01*i for i in range(78, 93)]
plt.plot(x, result[:len(result)], color='r', marker='o', linewidth=3,
         label="FLHDC-Binary")
plt.plot(x[:len(accuracy)], accuracy[:len(accuracy)], color='yellowgreen', linewidth=3,
         linestyle='-', label='FLHDC-SecureHD')
plt.legend()
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.xlabel('Retrain Epoch')
plt.ylabel('Accuracy')
plt.title("FLHDC Integer vs Binary")
plt.savefig(os.path.join(Dir, "dim10000_K20.png"))
plt.close()

''' Plot the accuracy of FLHDC_Binary vs Centralized HDC'''
centralized = []
Dir_cen = os.path.join(os.path.dirname(__file__),
                       '../..', 'Centralized HDC', 'Result')
with open(os.path.join(Dir_cen, 'retrain_60000+10000/Accuracy10000.csv'), 'r') as f:
    for line in f:
        # append the average of results for each parameter setup into the accuracy list

        centralized.append(np.array(line.strip().strip(
            ',').split(','), dtype=float))
# Get the accuracy of centralized HDC on ISOLET
centralized = np.array(centralized)
centralized = np.average(centralized, axis=0)

# Plot the accuracy

plt.plot(range(len(result)), result[:len(
    result)], color='r', marker='o', label="FLHDC-Binary")
plt.plot(range(len(centralized)), centralized, color='r', marker='o', linestyle='--',
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
