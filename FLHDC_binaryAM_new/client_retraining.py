import pandas as pd
import numpy as np
import os
import pickle
import library.HDC_FL_binary as HDC
'''
* Train the HDC Model for each client and retrain the Model afterward.
'''


def partition_train_val(train_data, train_label, ratio=0.8):
    ''' 
    * Partition local dataset into train/val sets to do retraining.
    * @ratio: the ratio of # of train_data to total data amount
    '''
    size_of_train_data = int(len(train_data)*ratio)
    size_of_val_data = len(train_data)-size_of_train_data

    # randomly select the index of validation data
    val_index = np.random.choice(
        [i for i in range(len(train_data))], size=size_of_val_data, replace=False)
    # select validation data from local dataset
    val_data = train_data[val_index]
    val_label = train_label[val_index]
    # delete validation data from local dataset to acquire train data
    train_data = np.delete(train_data, val_index, axis=0)
    train_label = np.delete(train_label, val_index, axis=0)
    return train_data, train_label, val_data, val_label


def load_data(client_number):
    '''load and return the data for each client'''
    client_path = os.path.join(os.path.dirname(
        __file__), 'client'+str(client_number))
    # the path of data(by label) for the client
    train_data_path = os.path.join(client_path, 'train_data.csv')
    train_label_path = os.path.join(client_path, 'train_label.csv')

    # load the csv file and transfer them into numpy array
    train_data = pd.read_csv(train_data_path)
    train_data = np.asarray(train_data.iloc[:, 1:])

    train_label = pd.read_csv(train_label_path)
    train_label = np.asarray(train_label.iloc[:, 1:])

    # check the size of each dataset
    print("Size of training data:{}".format(train_data.shape))
    # print("Size of training label:{}".format(train_label.shape))
    return train_data, train_label


def client_retraining(client_number, dimension, level, Nof_feature,
                      PCA_Projection, nof_class, CIM_vector, IM_vector, maximum, minimum, binaryAM):
    ''' 
    * Train HDC model for each client and save the AM and Size of Local 
    * Dataset as "Upload.pickle" for Gloabl model
    '''
    file_dir = os.path.dirname(__file__)
    train_data, train_label = load_data(client_number)
    # # Partition data into train/val dataset
    # train_data, train_label, val_data, val_label = partition_train_val(
    #     train_data=train_data, train_label=train_label)
    # Load the Global Model
    MNIST = HDC.HDC(dim=dimension, nof_class=nof_class,
                    nof_feature=Nof_feature, level=level, PCA_projection=PCA_Projection, binaryAM=binaryAM)
    MNIST.load_model(os.path.join(
        file_dir, 'Global_model', 'global_model_dict.pickle'))

    # Size of Mini-Batch
    ''' 
    * In retrain process, we randomly sample a batch of data to collect retrain_vector
    * so that we can prevent the oscillation of global model
    * (The uploaded retrain_vector are accumulated and binarized. As a result, 
    * too large size may cause the quality become bad )
    '''
    batch_size = np.minimum(
        len(train_data), (len(train_data))//5)
    batch_index = np.random.choice(
        range(len(train_data)), batch_size, replace=False)
    # Retrain the AM
    _, Times_add, Times_sub = MNIST.retrain(test_x=train_data[batch_index], test_y=train_label[batch_index], train_x=train_data[batch_index], train_y=train_label[batch_index], num_epoch=1, train_acc_demand=0.7, batch_size=batch_size, save_path=os.path.join(os.path.join(os.path.dirname(
        __file__), 'client'+str(client_number)), 'Retrain_Model.pickle'))

    # Save the Retrain_vector and Size of local Dataset as pickle file
    Upload_to_Server = {}
    for label in range(nof_class):
        Upload_to_Server['Times_add' +
                         str(label)] = Times_add[label]
        Upload_to_Server['Times_sub' +
                         str(label)] = Times_sub[label]
    Upload_to_Server['Retrain_vector_add'] = MNIST.retrain_vector_add
    Upload_to_Server['Retrain_vector_sub'] = MNIST.retrain_vector_sub
    with open(os.path.join(os.path.join(os.path.dirname(
            __file__), 'client'+str(client_number)), 'Upload.pickle'), 'wb') as f:
        pickle.dump(Upload_to_Server, f)


def main():
    # Read the setting of HDC Model from setup.pickle
    with open(os.path.join(os.path.dirname(__file__), 'Setup.pickle'), 'rb') as f:
        Base_model = pickle.load(f)
    nof_clients = int(Base_model['K'])
    dimension = int(Base_model['D'])
    level = int(Base_model['l'])
    Nof_feature = int(Base_model['Nof_feature'])
    PCA_Projection = bool(Base_model['PCA_Projection'])
    nof_class = int(Base_model['L'])
    CIM_vector = Base_model['CIM']
    IM_vector = Base_model['IM']
    maximum = Base_model['maximum']
    minimum = Base_model['minimum']
    binaryAM = Base_model['BinaryAM']
    print("{:=^40}".format("Client Retraining"))

    # Retrain Global Model by local dataset
    for client in range(1, nof_clients+1):
        print("{:-^40}".format("Client{}/{}").format(client, nof_clients))
        client_retraining(client, dimension, level, Nof_feature,
                          PCA_Projection, nof_class, CIM_vector, IM_vector, maximum, minimum, binaryAM)


if __name__ == "__main__":
    main()
