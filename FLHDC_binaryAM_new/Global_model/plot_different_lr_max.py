from matplotlib import rc
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib
'''
* Plot the accuracy of each parameter setup
'''
# FL-Binary AM
Dir = os.path.dirname(__file__)
# result: accuracy of FL_BinaryAM
FLHDC_binary_dim10000_K20 = {}
for lr_max in [3, 5, 7, 10, 20]:
    one_lr_max = []  # the result of a specified number of lr_max
    if lr_max == 5:
        with open(os.path.join(Dir, 'dim'+str(10000)+"_K"+str(20)+"_lr_retraininit.csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                if line.strip() != "":
                    one_lr_max.append(np.array(line.strip().strip(
                        ',').split(','), dtype=float))
            one_lr_max = np.average(one_lr_max, axis=0)

    else:
        with open(os.path.join(Dir, 'dim'+str(10000)+"_K"+str(20)+"_lr"+str(lr_max)+"_retraininit.csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                if line.strip() != "":
                    one_lr_max.append(np.array(line.strip().strip(
                        ',').split(','), dtype=float))
            one_lr_max = np.average(one_lr_max, axis=0)

    # Put the training process of specifed lr_max in the dict
    FLHDC_binary_dim10000_K20[lr_max] = one_lr_max[:31]
# Put the result of no learning rate
with open(os.path.join(Dir, 'dim'+str(10000)+"_K"+str(20)+"_nolr_retraininit.csv"), 'r') as f:
    one_lr_max = []
    for line in f:
                # append the average of results for each parameter setup into the accuracy list
        if line.strip() != "":
            one_lr_max.append(np.array(line.strip().strip(
                ',').split(','), dtype=float))
    one_lr_max = np.average(one_lr_max, axis=0)
    # Put the training process of specifed lr_max in the dict
    FLHDC_binary_dim10000_K20[1] = one_lr_max[:31]
# Put the result of no learning rate no retrain init
with open(os.path.join(Dir, 'dim'+str(10000)+"_K"+str(20)+"_lr1_noretraininit.csv"), 'r') as f:
    one_lr_max = []
    for line in f:
                # append the average of results for each parameter setup into the accuracy list
        if line.strip() != "":
            one_lr_max.append(np.array(line.strip().strip(
                ',').split(','), dtype=float))
    one_lr_max = np.average(one_lr_max, axis=0)
    # Put the training process of specifed lr_max in the dict
    FLHDC_binary_dim10000_K20[0] = one_lr_max[:31]


# Plot the epoch rounds vs Accuracy
# Each line indicates different lr_max

# activate latex text rendering
x_dim = range(31)
matplotlib.rcParams['font.style'] = 'italic'
# for lr_max in [0]:
#     plt.plot(x_dim,
#              FLHDC_binary_dim10000_K20[lr_max], marker='o', linestyle='--', label="Nolr_No_retrainint")
for lr_max in [1]:
    plt.plot(x_dim,
             FLHDC_binary_dim10000_K20[lr_max], marker='o', linestyle='--', label="lr=1 (constant)")

for lr_max in [3, 5,  20]:
    plt.plot(x_dim,
             FLHDC_binary_dim10000_K20[lr_max], marker='o', label="$\it{lr}$"+"$\it_{max}$"+"={} (Adaptive)".format(lr_max))
plt.legend()
plt.xlabel("Retraining Rounds")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(os.path.dirname(__file__),
                         "impact_of_lr_max.eps"), format='eps', bbox_inches='tight')
plt.show()
plt.close()
