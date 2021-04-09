
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
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+"_lr_retraininit.csv"), 'r') as f:
            for line in f:
                if not line.strip() == "":
                    # not an empty line
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
                if not line.strip() == "":
                    # not an empty line
                    secureHD.append(np.array(line.strip().strip(
                        ',').split(','), dtype=float))
secureHD = np.array(secureHD)
secureHD_full = np.average(secureHD, axis=0)[:33]
secureHD = np.average(secureHD, axis=0)[:len(result)]
# plot the accuracy between centralized_binary and FLHDC_binary
Dir3 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'Centralized HDC', 'Result', 'retrain_60000+10000')

# Load the accuracy of centralized HDC
centralized_df = pd.read_csv(os.path.join(Dir3, 'Accuracy10000.csv'))
centralized_df = centralized_df.iloc[:, 1:]
centralized_HD = np.array(centralized_df)
centralized_HD = np.average(centralized_df, axis=0)[:len(result)]

# FL BinaryHD on IID
FLHDC_binary_IID = []
Dir4 = os.path.join(os.path.dirname(__file__), '..', '..',
                    'FLHDC_binaryAM_new_IID', 'Global_model')
for K in [20]:
    for dim in [10000]:
        with open(os.path.join(Dir4, 'dim'+str(dim)+"_K"+str(K)+"_lr_retraininit.csv"), 'r') as f:
            for line in f:
                if not line.strip() == "":
                    # not an empty line
                    # append the average of results for each parameter setup into the accuracy list

                    FLHDC_binary_IID.append(np.array(line.strip().strip(
                        ',').split(','), dtype=float))
FLHDC_binary_IID = np.array(FLHDC_binary_IID)
FLHDC_binary_IID = np.average(FLHDC_binary_IID, axis=0)


# Plot the accuracy between Centralized HDC, SecureHD and FL BinaryHD
x = [i for i in range(len(secureHD))]
x_tick = [i for i in range(0, len(secureHD), 2)]
y = [0.01*i for i in range(81, 94)]
print(secureHD[-1]-result[-1])

plt.figure(figsize=(6.4, 4.8))
plt.plot(x, result, color='r', marker='o', markersize=4, linewidth=2,
         label="FL-HDC")
plt.plot(x, secureHD, color='yellowgreen', linewidth=2,
         linestyle='-', label='SecureHD')
# plt.plot(x, centralized_HD, color='r', marker='o', linestyle='--',
#          label="Centralized-Binary")
# plt.plot(x, FLHDC_binary_IID, color='b', marker='o', linewidth=3,
#          label="Proposed -IID")
plt.legend(loc='lower right')
plt.xticks(x_tick)
plt.yticks(y)
plt.grid()
plt.xlabel('Retraining Rounds')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(Dir, "Accuracy_framework.eps"),
            format='eps', bbox_inches='tight')
plt.show()
plt.close()

''' Performance versus Communication cost'''
# 因為全部accuracy都要畫的話圖片會太擠，所以我們就單純畫出特定幾個iteration的performance
# Compute the communication cost for some retrain iteration
cost_result = [round(math.log(((2*(10000+32)+10000)*10*i)), 1)
               for i in range(1, len(result)+1)]
# cost_result = [round(((2*(10000+32)+10000)*10*i), 1)
#                for i in range(1, len(result)+1)]
cost_secureHD = [round(math.log((10000*32*2)*10*i), 1)
                 for i in range(1, len(secureHD_full)+1)]
# cost_secureHD = [round(((10000*32*2)*10*i), 1)
#  for i in range(1, len(secureHD_full)+1)]
# Print out those points whose accuracy is larger than 0.88
for i in range(len(cost_result)):
    if result[i] > 0.88:
        print(i, (2*(10000+32)+10000)*10*(i+1), cost_result[i])
        break
for i in range(len(cost_secureHD)):
    if secureHD_full[i] > 0.88:
        print(i, (10000*32*2)*10*(i+1),  cost_secureHD[i])
        break
print("Communication cost ration(acc>0.88):{}".format(
    ((2*(10000+32)+10000)*10*(13+1)) / ((10000*32*2)*10*(5+1))))

# The xticks to drawn
x_result = [cost_result[i]
            for i in [0, 5, 13, 25]]
x_secureHD = [cost_secureHD[i]
              for i in [5,  10, 20, len(secureHD_full)-1]]
x = x_result + x_secureHD
y = [0.01 * i for i in range(81,  94)]

# select partial points
points_chosen_result = [i for i in [0, 5, 10, 13, 15, 20, 25, 30]]
points_chosen_SecureHD = [i for i in [
    0, 5,  10, 20, 25, 30, len(secureHD_full)-1]]

# select accuracy for partial retrain iteration since drawing all the points are too messy
result = [result[i] for i in points_chosen_result]
secureHD_full = [secureHD_full[i] for i in points_chosen_SecureHD]
cost_result = [cost_result[i] for i in points_chosen_result]
cost_secureHD = [cost_secureHD[i] for i in points_chosen_SecureHD]

plt.figure(figsize=(6.4, 4.8))
plt.plot(cost_result, result, color='r',  linewidth=3, marker='o',
         label="Proposed FL-HDC")
plt.plot(cost_secureHD, secureHD_full, color='yellowgreen',  marker='o', linewidth=3,
         linestyle='-', label='SecureHD[11]')
plt.hlines(0.88, cost_result[0], cost_secureHD[-1]+0.5,
           linestyles='--', linewidth=5)    # Plot the 0.88 accuracy line
# Plot the vertical line at where our proposed framework is above 0.88
# plt.vlines(14.8, 0.78, 0.88, colors='r', linestyles='--', linewidth=3)
# # Plot the vertical line at where SecureHD achieves 0.88
# plt.vlines(17.5, 0.78, 0.88, colors='yellowgreen',
#            linestyles='--', linewidth=3)

plt.legend(loc='lower right')
plt.xticks(x)
plt.xlim(x[0], x[-1]+0.3)
plt.yticks(y)
plt.ylim(0.81, y[-1])
plt.grid()
plt.xlabel('Communication Cost(log)')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(Dir, "Communication_cost.eps"),
            format='eps', bbox_inches='tight')
plt.show()
plt.close()
