import pandas as pd
import numpy as np
import os
import pickle
import HDC_mulpc_ISOLET as HDC


def load_data(client_number):
    '''load and return the data for each client'''
    client_path = 'client'+str(client_number)
    # the path of data(by label) for the client
    train_data_path = os.path.join(client_path, 'train_data.csv')
    train_label_path = os.path.join(client_path, 'train_label.csv')
    test_data_path = os.path.join(client_path, 'test_data.csv')
    test_label_path = os.path.join(client_path, 'test_label.csv')

    # load the csv file and transfer them into numpy array
    train_data = pd.read_csv(train_data_path)
    train_data = np.asarray(train_data.iloc[:, 1:])

    train_label = pd.read_csv(train_label_path)
    train_label = np.asarray(train_label.iloc[:, 1:])

    test_data = pd.read_csv(test_data_path)
    test_data = np.asarray(test_data.iloc[:, 1:])

    test_label = pd.read_csv(test_label_path)
    test_label = np.asarray(test_label.iloc[:, 1:])

    # check the size of each dataset
    print("Size of training data:{}".format(train_data.shape))
    # print("Size of training label:{}".format(train_label.shape))
    print("Size of testing data:{}".format(test_data.shape))
    # print("Size of testing label:{}".format(test_label.shape))
    return train_data, train_label, test_data, test_label


def client_training(client_number, dimension, level, Nof_feature,
                    PCA_Projection, nof_class, CIM_vector, IM_vector):
    ''' 
    * Train HDC model for each client and save the AM and Size of Local 
    * Dataset as "Upload.pickle" for Gloabl model
    '''
    train_data, train_label, test_data, test_label = load_data(client_number)
    MNIST = HDC.HDC(dim=dimension, nof_class=nof_class,
                    nof_feature=Nof_feature, level=level, PCA_projection=PCA_Projection)
    MNIST.train(train_data, train_label,
                IM_vector=IM_vector, CIM_vector=CIM_vector)
    # Save the AM and Size od local Dataset as pickle file
    Upload_to_Server = {}
    Upload_to_Server['Size'] = len(train_data)
    Upload_to_Server['AM'] = MNIST.Prototype_vector
    with open(os.path.join('client'+str(client_number), 'Upload.pickle'), 'wb') as f:
        pickle.dump(Upload_to_Server, f)


def main():
    # Read the setting of HDC Model from setup.pickle
    with open('./Setup.pickle', 'rb') as f:
        Base_model = pickle.load(f)
    nof_clients = int(Base_model['K'])
    dimension = int(Base_model['D'])
    level = int(Base_model['l'])
    Nof_feature = int(Base_model['Nof_feature'])
    PCA_Projection = bool(Base_model['PCA_Projection'])
    nof_class = int(Base_model['L'])
    CIM_vector = Base_model['CIM']
    IM_vector = Base_model['IM']
    print("{:=^40}".format("The setting of FL HDC"))
    print("Number of clients:{}".format(nof_clients))
    print("Dimension:{}".format(dimension))
    print("Level:{}".format(level))
    print("Nof_feature:{}".format(Nof_feature))
    print("PCA_Projection:{}".format(PCA_Projection))
    print("Nof_class:{}".format(nof_class))
    print("Size of CIM:({}, {}, {})".format(len(CIM_vector),
                                            CIM_vector[0].shape[0], CIM_vector[0].shape[1]))
    print("Size of IM:({}, {}, {})".format(len(IM_vector),
                                           IM_vector['feature1'].shape[0], IM_vector['feature1'].shape[1]))
    print("{:=^40}".format("Begin training"))

    for client in range(1, nof_clients+1):
        print("{:-^40}".format("Client{}/{}").format(client, nof_clients))
        client_training(client, dimension, level, Nof_feature,
                        PCA_Projection, nof_class, CIM_vector, IM_vector)


if __name__ == "__main__":
    main()
