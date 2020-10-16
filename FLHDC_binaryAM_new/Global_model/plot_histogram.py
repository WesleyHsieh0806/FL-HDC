import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

Dir = os.path.dirname(__file__)
if not os.path.isdir(os.path.join(Dir, 'histogram')):
    os.makedirs(os.path.join(Dir, 'histogram'))
# Load the Binary AM of One-shot FLHDC_BInary
with open(os.path.join(Dir, 'global_model_dict.pickle'), 'rb') as f:
    FLHDC_binary = pickle.load(f)
# Check the size of class 1 hv
FL_integerAM = FLHDC_binary['Prototype_vector']['integer']
print("{:=^40}".format("Plot Histogram"))
print(FL_integerAM[1].shape)
print("maximum:{} minimum:{}".format(
    np.max(FL_integerAM[1]), np.min(FL_integerAM[1])))
# Plot the histogram of class hv
for label in range(1, 3):
    # FLHDC
    plt.hist(FL_integerAM[label][0], range=[-7000, 7000], bins=10, color='red')
    plt.title("FL: class "+str(label)+" hv")
    plt.xlabel("Value Distribution")
    # Save the figure
    if len(sys.argv) > 1:
        plt.savefig(os.path.join(Dir, 'histogram',
                                 'Retrain{}_FL_hist_'.format(sys.argv[1])+str(label)+'.png'))
    else:
        plt.savefig(os.path.join(Dir, 'histogram',
                                 'One-shot_FL_hist_'+str(label)+'.png'))
    plt.close()
