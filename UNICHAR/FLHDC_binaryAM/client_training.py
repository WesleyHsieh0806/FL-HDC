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
    # print("Size of testing label:{}".format(test_label.shape))
    return train_data, train_label


def client_training(client_number, dimension, level, Nof_feature,
                    PCA_Projection, nof_class, CIM_vector, IM_vector, maximum, minimum, binaryAM):
    ''' 
    * Train HDC model for each client and save the AM and Size of Local 
    * Dataset as "Upload.pickle" for Gloabl model
    '''
    train_data, train_label = load_data(client_number)

    # Train and Retrain the Model
    UNICHAR = HDC.HDC(dim=dimension, nof_class=nof_class,
                      nof_feature=Nof_feature, level=level, PCA_projection=PCA_Projection, binaryAM=binaryAM)
    UNICHAR.train(train_data, train_label,
                  IM_vector=IM_vector, CIM_vector=CIM_vector, maximum=maximum, minimum=minimum)

    # Save the AM and Size of local Dataset as pickle file
    Upload_to_Server = {}
    for label in range(1, nof_class+1):
        Upload_to_Server['Size' +
                         str(label)] = np.count_nonzero(train_label == label)
    Upload_to_Server['AM'] = UNICHAR.Prototype_vector
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
    print("{:=^40}".format("The setting of FL HDC"))
    print("Number of clients:{}".format(nof_clients))
    print("Dimension:{}".format(dimension))
    print("Level:{}".format(level))
    print("Nof_feature:{}".format(Nof_feature))
    print("PCA_Projection:{}".format(PCA_Projection))
    print("Binary AM:{}".format(binaryAM))
    print("Nof_class:{}".format(nof_class))
    print("Size of CIM:({}, {}, {})".format(len(CIM_vector),
                                            CIM_vector[0].shape[0], CIM_vector[0].shape[1]))
    print("Size of IM:({}, {}, {})".format(len(IM_vector),
                                           IM_vector['feature1'].shape[0], IM_vector['feature1'].shape[1]))
    print("{:=^40}".format("Begin training"))

    for client in range(1, nof_clients+1):
        print("{:-^40}".format("Client{}/{}").format(client, nof_clients))
        client_training(client, dimension, level, Nof_feature,
                        PCA_Projection, nof_class, CIM_vector, IM_vector, maximum, minimum, binaryAM)


if __name__ == "__main__":
    main()
