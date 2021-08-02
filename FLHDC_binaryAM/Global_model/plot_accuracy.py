import math
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
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+"_nolr_noretraininit.csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                result.append(np.array(line.strip().strip(
                    ',').split(','), dtype=float))
result = np.array(result)
result = np.average(result, axis=0)
# FL SecureHD
Dir2 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'FLHDC_IntegerAM', 'Global_model')
secureHD = []
for K in [20]:
    for dim in [10000]:
        with open(os.path.join(Dir2, 'dim'+str(dim)+"_K"+str(K)+".csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list

                secureHD.append(np.array(line.strip().strip(
                    ',').split(','), dtype=float))
secureHD = np.array(secureHD)
secureHD = np.average(secureHD, axis=0)[:len(result)]

# plot the accuracy between centralized_binary and FLHDC_binary
Dir3 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'Centralized HDC', 'Result', 'retrain_60000+10000')

# Load the accuracy of centralized HDC
centralized_df = pd.read_csv(os.path.join(Dir3, 'Accuracy10000.csv'))
centralized_df = centralized_df.iloc[:, 1:]
centralized_HD = np.array(centralized_df)
centralized_HD = np.average(centralized_df, axis=0)[:len(result)]
# Plot the accuracy between Centralized HDC, SecureHD and FL BinaryHD
x = [i for i in range(len(secureHD))]
y = [0.01*i for i in range(78, 92)]
# naive: retrain without retrain vectors
naive = [result[0] for i in range(len(result))]

plt.plot(x, result, color='r', marker='o', linewidth=3,
         label="Proposed FLHDC-Binary")
plt.plot(x, secureHD, color='yellowgreen', linewidth=3,
         linestyle='-', label='FLHDC-SecureHD')
plt.plot(x, centralized_HD, color='r', marker='o', linestyle='--',
         label="Centralized-Binary")
plt.plot(x, naive, color='k', marker='o', linewidth=3,
         label="Naive")
plt.legend()
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.xlabel('Retrain Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy")
plt.savefig(os.path.join(Dir, "dim10000_K20.png"))
plt.close()

''' Performance versus Communication cost'''
# 因為全部accuracy都要畫的話圖片會太擠，所以我們就單純畫出特定幾個iteration的performance
# Compute the communication cost for some retrain iteration
cost_result = [round(math.log(((10000+32+10000)*10*i)), 1)
               for i in range(1, len(result)+1, 5)]
cost_secureHD = [round(math.log((10000*32*2)*10*i), 1)
                 for i in range(1, len(result)+1, 5)]
# select accuracy for partial retrain iteration
result = [result[i] for i in range(0, 31, 5)]
secureHD = [secureHD[i] for i in range(0, 31, 5)]

plt.plot(cost_result, result, color='r', marker='o', linewidth=3,
         label="Proposed FLHDC-Binary")
plt.plot(cost_secureHD, secureHD, color='yellowgreen', marker='o', linewidth=3,
         linestyle='-', label='FLHDC-SecureHD')

cost_result = [cost_result[i]
               for i in [0, 1, 2, 4]]
cost_secureHD = [cost_secureHD[i]
                 for i in [0,  3,  5]]
x = cost_result + cost_secureHD

plt.legend()
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.xlabel('Communication Cost (log)')
plt.ylabel('Accuracy')
plt.title("Accuracy")
plt.savefig(os.path.join(Dir, "dim10000_K20_communication_cost.png"))
plt.close()
