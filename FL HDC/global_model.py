import pandas as pd
import os
import numpy as np
import HDC_mulpc_ISOLET as HDC
import pickle
# the global model needs CIM IM AM maximum minimum difference level dimension
# nof_class to acquire test-set accuracy


def main():
    ''' Load "Setup.pickle" to acquire the number of clients '''
    with open('./Setup.pickle', 'rb') as f:
        Base_model = pickle.load(f)
    nof_clients = int(Base_model['K'])
    ''' load the size of local dataset and the AM from each client'''
    Prototype_vector = {}
    for client in range(1, nof_clients+1):
        with open(os.path.join('client'+str(client), 'Upload.pickle'), 'rb') as f:
            # the size of local dataset and AM are included in Upload.pickle
            client_dict = pickle.load(f)
        print(client_dict['Size'])


if __name__ == "__main__":
    main()
