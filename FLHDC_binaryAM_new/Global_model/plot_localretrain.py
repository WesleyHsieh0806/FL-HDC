import matplotlib.pyplot as plt
import os
import numpy as np
''' 
* For the Final presentation
* To solve the fluctuation
* We have to compare the difference between local retrain/without local retrain
'''
Dir = os.path.dirname(__file__)

with open(os.path.join(Dir, '..', '..', 'FLHDC_binaryAM', 'Global_model', 'dim'+str(10000)+"_K"+str(20)+"_nolr_noretraininit.csv"), 'r') as f:
    one_lr_max = []
    for line in f:
                # append the average of results for each parameter setup into the accuracy list
        if line.strip() != "":
            one_lr_max.append(np.array(line.strip().strip(
                ',').split(','), dtype=float))
    one_lr_max = np.average(one_lr_max, axis=0)
    # Put the training process of specifed lr_max in the dict
    no_localretrain = one_lr_max[:31]
with open(os.path.join(Dir, 'dim'+str(10000)+"_K"+str(20)+"_nolr_retraininit.csv"), 'r') as f:
    one_lr_max = []
    for line in f:
                # append the average of results for each parameter setup into the accuracy list
        if line.strip() != "":
            one_lr_max.append(np.array(line.strip().strip(
                ',').split(','), dtype=float))
    one_lr_max = np.average(one_lr_max, axis=0)
    # Put the training process of specifed lr_max in the dict
    local_retrain_nolr = one_lr_max[:31]

# Plot localretrain vs no localretrain
plt.plot(local_retrain_nolr, color='b', marker='o', label='Local Retrain')
plt.plot(no_localretrain, color='b', marker='o',
         linestyle='--', label='Previous')
plt.legend()
plt.grid()
plt.xlabel('Retrain Rounds')
plt.ylabel('Accuracy')
plt.show()
