import pandas as pd
import os
import numpy as np
from library import HDC_FL_binary as HDC
import pickle
import sys
'''
* Aggregate the AM of each client Model to acquire global Model 
* Report the test-set accuracy of Global Model and save them as csv file
* The only difference between this file and "global_model.py" is the path of saved csv files
'''


def load_data():
    '''Load test dataset '''
    sort_test_data_path = os.path.join(os.path.dirname(
        __file__), '../../MNIST/sorted_data_csv/test_data.csv')
    sort_test_label_path = os.path.join(os.path.dirname(
        __file__), '../../MNIST/sorted_data_csv/test_label.csv')

    # load the csv file and transfer them into numpy array
    sort_test_data = pd.read_csv(sort_test_data_path)
    sort_test_data = np.asarray(sort_test_data.iloc[:, 1:])

    sort_test_label = pd.read_csv(sort_test_label_path)
    sort_test_label = np.asarray(sort_test_label.iloc[:, 1:])

    # check the size of each dataset
    print("Size of testing data:{}".format(sort_test_data.shape))
    print("Size of testing label:{}".format(sort_test_label.shape))
    return sort_test_data, sort_test_label


def main():
    file_dir = os.path.dirname(__file__)
    print("{:=^40}".format("Global Model"))
    ''' 
    * Load "Setup.pickle" to acquire the parameter setup 
    '''
    with open(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'Setup.pickle'), 'rb') as f:
        Base_model = pickle.load(f)
    # load the parameters
    nof_clients = int(Base_model['K'])
    dimension = int(Base_model['D'])
    level = int(Base_model['l'])
    Nof_feature = int(Base_model['Nof_feature'])
    PCA_Projection = bool(Base_model['PCA_Projection'])
    BinaryAM = bool(Base_model['BinaryAM'])
    nof_class = int(Base_model['L'])
    CIM_vector = Base_model['CIM']
    IM_vector = Base_model['IM']
    maximum = Base_model['maximum']
    minimum = Base_model['minimum']
    difference = maximum-minimum+1e-8
    ''' load the size of local dataset and the AM from each client'''
    Prototype_vector = {}
    Prototype_vector['binary'] = {}
    Prototype_vector['integer'] = {}
    # case1: training
    if len(sys.argv) == 1:
        # Weighted Average the class hypervector by local data size
        for client in range(1, nof_clients+1):
            with open(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'client'+str(client)), 'Upload.pickle'), 'rb') as f:
                # the size of local dataset and AM are included in Upload.pickle
                client_dict = pickle.load(f)
            if client == 1:
                for label in range(nof_class):
                    # Ck = sigma(njk * Ckj) @njk:class-k data size from clientj
                    # Ckj: class-k hypervector from client j
                    size = client_dict['Size'+str(label)]
                    Prototype_vector['integer'][label] = size * \
                        client_dict['AM']['binary'][label]
            else:
                # Ck = sigma(njk * Ckj) @njk:class-k data size from clientj
                # Ckj: class-k hypervector from client j
                for label in range(nof_class):
                    size = client_dict['Size'+str(label)]
                    Prototype_vector['integer'][label] += size * \
                        client_dict['AM']['binary'][label]
    # case2: retrain
    else:
        ''' 
        * In retraining phase, the IntegerAM should be updated using the uploaded binarized retrain_vectors
        * as this formula: Ck = Ck + sigma(njk * Rkj)  (Rkj:retrain_vectors of class-k from client j) 
        '''
        # Read the IntegerAM of Global Model to do retrain
        with open(os.path.join(file_dir, 'global_model_dict.pickle'), 'rb') as f:
            last_global_model = pickle.load(f)
        Prototype_vector = last_global_model['Prototype_vector']

        # We have sent an argument "retrain_epoch" into global_model.py to differentiate it from training process
        for client in range(1, nof_clients+1):
            with open(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'client'+str(client)), 'Upload.pickle'), 'rb') as f:
                # the size of local dataset and AM are included in Upload.pickle
                client_dict = pickle.load(f)
            for label in range(nof_class):
                size = client_dict['Size'+str(label)]
                Prototype_vector['integer'][label] += size * \
                    client_dict['Retrain_vector'][label]

    ''' Binarize it to acquire binary AM'''
    for CLASS in range(0, nof_class):
        # After Retraining, the binary Prototype vector should be updated()
        # As a result, we have to binarize them (>0 --> 1   <0 --> -1)
        # Special case: if an element is 0, then randomly change it into 1 or -1
        if len(sys.argv) != 1:
            last_vector = Prototype_vector['binary'][CLASS]

        Prototype_vector['binary'][CLASS] = np.zeros(
            Prototype_vector['integer'][CLASS].shape).astype(int)
        Prototype_vector['binary'][CLASS][Prototype_vector['integer'][CLASS]
                                          > 0] = 1
        Prototype_vector['binary'][CLASS][Prototype_vector['integer'][CLASS]
                                          < 0] = -1
        Prototype_vector['binary'][CLASS][Prototype_vector['integer'][CLASS] == 0] = np.random.choice(
            [1, -1], size=np.count_nonzero(Prototype_vector['integer'][CLASS] == 0))
        print("# of 0:", np.count_nonzero(
            Prototype_vector['integer'][CLASS] == 0))
        if len(sys.argv) != 1:
            print("Flipped bit:", np.count_nonzero(
                Prototype_vector['binary'][CLASS] != last_vector))
    ''' Print out the current iteration'''
    if len(sys.argv) > 1:
        print("Retrain Epoch:{}".format(sys.argv[1]))
        # We will send a parameter "retrain", which means the global model is in retraining phase

    '''Save the global model parameters as pickle files'''
    global_model_dict = {}
    global_model_dict['nof_dimension'] = dimension
    global_model_dict['nof_class'] = nof_class
    global_model_dict['nof_feature'] = Nof_feature
    global_model_dict['level'] = level
    global_model_dict['IM'] = IM_vector
    global_model_dict['CIM'] = CIM_vector
    global_model_dict['Prototype_vector'] = Prototype_vector
    global_model_dict['max'] = maximum
    global_model_dict['min'] = minimum
    global_model_dict['difference'] = difference
    global_model_dict['PCA'] = PCA_Projection
    global_model_dict['BinaryAM'] = BinaryAM
    with open(os.path.join(file_dir, 'global_model_dict.pickle'), 'wb') as f:
        pickle.dump(global_model_dict, f)

    ''' Use Global model to report test-set accuracy'''
    MNIST = HDC.HDC(dimension, nof_class, Nof_feature, level)
    MNIST.load_model(os.path.join(file_dir, 'global_model_dict.pickle'))
    ''' load test dataset'''
    test_data, test_label = load_data()
    y_pred = MNIST.test(test_x=test_data)
    acc = MNIST.accuracy(y_true=test_label, y_pred=y_pred)
    print("Accuracy:{:.4f}".format(acc))

    with open(os.path.join(file_dir, 'dim'+str(dimension)+"_K"+str(nof_clients)+'.csv'), 'a') as f:
        if len(sys.argv) == 1:
            f.write('\n')
        f.write(str(acc)+',')


if __name__ == "__main__":
    main()
