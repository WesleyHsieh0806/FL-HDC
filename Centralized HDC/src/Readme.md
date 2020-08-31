# Introduction
Centralized HDC on MNIST Dataset
There are two experiments here:
*1. Centralized HDC on Total IID MNIST Dataset*
*2. Cetralized HDC+Retrain on Total IID MNIST Dataset*

# Description about the parameter setup of HDC:
Dimension:[1000,2000,5000,10000]
retrain_epoch:30
Encoded Hypervector: Integer
AM:Binary
CIM_Level:21

# File Description
## retrain_centralized_HDC.py:
Execute the retraining with each parameter setup for HDC Model multiple times
Record the retrain accuracy and execution time for each epoch
## centralized_HDC.py:
Execute the state-of-art one-shot HDC on MNIST dataset with each parameter setup multiple times
Record the average accuracy and execution time
## HDC_Centralized.py:
The library of HDC Model
## HDC_model.pickle:
The best model saved in retraining progress
