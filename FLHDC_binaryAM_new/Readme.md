# FLHDC_BinaryAM_new:
The files in this directory are implementation of the Final Version of FLHDC with BinaryAM and Retraining Updates.
## Usage:
```bash
$ python "Base Model/Base_model.py"
$ python client_training.py
$ python Global_model/global_model.py 
$ python client_retraining.py.py 
$ python Global_model/global_model.py [retrain_epoch]
```
or
```bash
$ python Total_retraining.py
```

## Functionality

### **Base_model.py**

*  Setup the parameters and CIM, IM vectors for federated learning.

*  **Output**: "Setup.pickle"
### **client_training.py:**

* Related training process with local dataset will be executed in this file

* **Output:**"Upload.pickle" The size of local dataset and the AM will be included in "Upload.pickle"

### **client_retraining.py:**

*  Much the same as client_training.py, except that client_retraining.py will partition local dataset into train and val sets to do retraining.


### **global_model.py:**

*  Do weighted addition to acquire the AM of global model like the formula below:
  
  
*  Then, return the test-set accuracy.
  
### **client_rmtree.py**

* delete all the directories of clients 

