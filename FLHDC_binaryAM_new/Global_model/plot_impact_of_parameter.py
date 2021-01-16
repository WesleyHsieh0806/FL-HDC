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
FLHDC_binary = {}
for K in [20, 60, 80, 100]:
    FLHDC_binary[K] = []
for K in [20, 60, 80, 100]:
    for dim in [1000, 2000, 4000, 6000, 8000, 10000]:
        oneK_dim_result = []
        with open(os.path.join(Dir, 'dim'+str(dim)+"_K"+str(K)+"_lr_retraininit.csv"), 'r') as f:
            for line in f:
                # append the average of results for each parameter setup into the accuracy list
                if line.strip() != "":
                    oneK_dim_result.append(np.array(line.strip().strip(
                        ',').split(','), dtype=float))
            oneK_dim_result = np.average(oneK_dim_result, axis=0)[30]

            # Append the result after 30 retrain rounds into the Total LIst
            FLHDC_binary[K].append(oneK_dim_result)
# Plot the accuracy vs Dimension
# Each line indicates different number of clients
x_dim = [1000, 2000, 4000, 6000, 8000, 10000]
for K in [20, 60, 80, 100]:
    print(FLHDC_binary[K])
    plt.plot(x_dim, FLHDC_binary[K], marker='o', label="K={}".format(K))
plt.legend()
plt.xlabel("Dimension")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(os.path.dirname(__file__),
                         "impact_of_parameter.eps"), format='eps', bbox_inches='tight')
plt.show()
plt.close()
