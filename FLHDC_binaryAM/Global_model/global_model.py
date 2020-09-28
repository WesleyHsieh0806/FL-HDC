import pandas as pd
import os
import numpy as np
from library import HDC_FL_binary as HDC
import pickle
import sys
import copy
'''
* Aggregate the AM of each client Model to acquire global Model 
* Report the test-set accuracy of Global Model and save them as csv file
* The only difference between this file and "global_model.py" is the path of saved csv files
'''


def new_lr(larger, equal, current_lr):
    ''' Modify the learning rate according to the relationship between # of flipped bit and 10
    @ Return: the value of the next learning rate
    '''
    new_lr = 0
    lr_list = list(range(1, 6))
    if larger:
        # Increase the lr to the one-level larger one
        level = lr_list.index(current_lr)
        level += 1
        if level >= len(lr_list):
            level = len(lr_list)-1
        new_lr = lr_list[level]
    elif equal:
        new_lr = current_lr
    else:
        # Decrease the lr to the one-level smaller one
        level = lr_list.index(current_lr)
        level -= 1
        if level < 0:
            level = 0
        new_lr = lr_list[level]
    return new_lr


def LR_Decider(flipped_bit, learning_rate, number_of_class):
    ''' Adjust the learning rate by # of flipped bit last time
    * the learning rate will lie in these given values [1, 2, 3, 4, 5] 
    * The intution here is to maintain the # of flipped bit to be around 10
    * If # of flipped_bit >10 : the learning rate will get one-level smaller: e.g. 2->1
    * If # of flipped_bit <10 : the learning rate will get one-level larger: e.g. 1->2
    * If # of flipped_bit =10 : the learning rate remains the same
    @ Return: A dictionary consists of new learning rate(learning_rate[Class] = XX)
    '''
    for Class in range(number_of_class):
        if flipped_bit[Class] > 10:
            lr_larger = False
            equal = False
            learning_rate[Class] = new_lr(
                lr_larger, equal, learning_rate[Class])
        elif flipped_bit[Class] == 10:
            lr_larger = False
            equal = True
            learning_rate[Class] = new_lr(
                lr_larger, equal, learning_rate[Class])
        else:
            lr_larger = True
            equal = False
            learning_rate[Class] = new_lr(
                lr_larger, equal, learning_rate[Class])
    return learning_rate


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
        * as this formula: Ck = Ck + sigma(learning rate * njk * Rkj)  (Rkj:retrain_vectors of class-k from client j) 
        '''
        # Read the IntegerAM of Global Model to do retrain
        with open(os.path.join(file_dir, 'global_model_dict.pickle'), 'rb') as f:
            last_global_model = pickle.load(f)
        '''
        # Note that we should use deep copy to prevent last_global_model to be changed 
        # since shallow copy will only copy first-level properties for a dictionary e.g. integer
        # while deep copy copies deep properties such as list and arrays'''
        Prototype_vector = copy.deepcopy(last_global_model['Prototype_vector'])

        # Record the total times of modification for each label
        Total_times = {}
        for Class in range(nof_class):
            Total_times[Class] = 0
        ''' Read the number of flipped bit and learning rate for each class 
        last time to determine the learning rate
        '''
        if os.path.isfile(os.path.join(file_dir, 'flipped_bit.pickle')):
            # If we have the record of flipped bit last time, determine the learning rate by it
            with open(os.path.join(file_dir, 'flipped_bit.pickle'), 'rb') as f:
                flipped_bit = pickle.load(f)
            with open(os.path.join(file_dir, 'learning_rate.pickle'), 'rb') as f:
                learning_rate = pickle.load(f)
            learning_rate = LR_Decider(
                flipped_bit, learning_rate, number_of_class=nof_class)
        else:
            # If we don't have the record last time, just assign the learning rate to be the largest level
            learning_rate = {}
            for Class in range(nof_class):
                learning_rate[Class] = 5

        # We have sent an argument "retrain_epoch" into global_model.py to differentiate it from training process
        for client in range(1, nof_clients+1):
            with open(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'client'+str(client)), 'Upload.pickle'), 'rb') as f:
                # the size of local dataset and AM are included in Upload.pickle
                client_dict = pickle.load(f)
            for label in range(nof_class):
                size = client_dict['Size'+str(label)]
                Prototype_vector['integer'][label] += int(learning_rate[label] * size) * \
                    client_dict['Retrain_vector'][label]
                # add the size to total modification times
                Total_times[label] += size
                # test the quality of class1 retrain_vector
                if label == 1:
                    print("# of different bits between class1 hv and gradient:{}".format(np.count_nonzero(
                        client_dict['Retrain_vector'][label] != Prototype_vector['binary'][label])))
        # Print out the total modification times
        for Class in range(nof_class):
            print("Class:{} Total times of modification:{}".format(
                Class, Total_times[Class]))

    ''' Binarize it to acquire binary AM'''
    flipped_bit = {}
    for CLASS in range(0, nof_class):
        # After Retraining, the binary Prototype vector should be updated()
        # As a result, we have to binarize them (>0 --> 1   <0 --> -1)
        # Special case: if an element is 0, then randomly change it into 1 or -1
        if len(sys.argv) != 1:
            last_vector = Prototype_vector['binary'][CLASS]

        Prototype_vector['binary'][CLASS] = np.zeros(
            Prototype_vector['integer'][CLASS].shape).astype(int)
        Prototype_vector['binary'][CLASS][Prototype_vector['integer'][CLASS]
                                          >= 0] = 1
        Prototype_vector['binary'][CLASS][Prototype_vector['integer'][CLASS]
                                          < 0] = -1
        print("# of 0:", np.count_nonzero(
            Prototype_vector['integer'][CLASS] == 0))
        # print out the number of flipped bit
        if len(sys.argv) != 1:
            number_of_flipped_bit = np.count_nonzero(
                Prototype_vector['binary'][CLASS] != last_vector)
            print("Flipped bit:", number_of_flipped_bit)
            print("Learning rate:", learning_rate[CLASS])
            # Save the number of flipped bit into dictionary
            flipped_bit[CLASS] = number_of_flipped_bit

            '''
            * To prevent the # of flipped bit explodes, we cancel the update of integerAM and binaryAM
            * once the # of flipped bit exceeds 1000
            '''
            if number_of_flipped_bit > 500:
                Prototype_vector['binary'][CLASS] = last_vector
                Prototype_vector['integer'][CLASS] = last_global_model['Prototype_vector']['integer'][CLASS]
                print(
                    "Flipped bit explodes!!! Cancel the update of Class {}".format(CLASS))
    ''' Print out the current iteration'''
    if len(sys.argv) > 1:
        # We will send a parameter "retrain", which means the global model is in retraining phase
        print("Retrain Epoch:{}".format(sys.argv[1]))
        print("Save the Flipped bit and Learning rate...")
        ''' Save the learning rate and flipped bit into pickle files'''
        with open(os.path.join(file_dir, 'flipped_bit.pickle'), 'wb') as f:
            pickle.dump(flipped_bit, f)
        with open(os.path.join(file_dir, 'learning_rate.pickle'), 'wb') as f:
            pickle.dump(learning_rate, f)

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

    with open(os.path.join(file_dir, 'dim'+str(dimension)+"_K"+str(nof_clients)+'_lr_smallbatch.csv'), 'a') as f:
        if len(sys.argv) == 1:
            f.write('\n')
        f.write(str(acc)+',')


if __name__ == "__main__":
    main()
