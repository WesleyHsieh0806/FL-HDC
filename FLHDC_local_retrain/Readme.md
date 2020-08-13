# Description about the functionality of each file
## Usage:
```bash
$ cd "./Base Model"
$ python "./Base_model.py"
$ cd ..
$ python client_training.py
$ python ./Global_model/global_model.py 
```
## ./Base Model/Base_model.py:
Setup the parameters and CIM, IM vectors for federated learning.
Since the models from all clients should follow these parameters, I call this setting "Base Model".
After setting the parameters, the directories for each client will be constructed and then the NonIID dataset for each client, IID dataset for global model initialization will be assigned.

***Output: After running this file, "Setup.pickle" will be made so that each model can follow the information in "Setup.pickle" to setup their models.***
## client_training.py:
After acquiring "Setup.pickle", we should train the model from local dataset of each client based on the setting of "Setup.pickle".
Related training process will be executed in this file

***Output:"Upload.pickle" The size of local dataset and the AM will be included in "Upload.pickle" for the usage of global model***

## client_retraining.py:
Much the same as client_training.py, except that client_retraining.py will partition local dataset into train and val sets to do retraining.

***Output:"Upload.pickle" The size of local dataset and the AM will be included in "Upload.pickle" for the usage of global model***

## global_model.py:
This file will do weighted addition to acquire the AM of global model like the formula below:
**$$ C_k = Σ^{K}_{j=1}n_j * C_{kj} $$**
$$ K= number\ of\ clients$$
$$ j = client\ index$$
Then, the global model will return the test-set accuracy.
## client_rmtree.py
delete all the directories of clients 
