from math import floor
import pandas as pd
import numpy as np
import pickle
import os
import argparse


def partition_initial_client(train_data, train_label):
    '''partition the dataset into two parts: 1.initial data for global model 2.client dataset'''
    # initial train dataset(initial_train_data):  234images, 9images for each label
    # client train dataset(train_data): 5978
    # client test dataset: 1559
    for label in range(1, 27):
        # train:for each label, randomly select 10 images and append it to the initial_train dataset
        # replace:False --> Will not place the element(which has been selected)back to the array
        select_index = np.random.choice(
            np.where(train_label.flatten() == label)[0], size=9, replace=False)
        if label == 1:
            initial_train_data = train_data[select_index]
            initial_train_label = train_label[select_index]
        else:
            # append the data by np.concatenate
            initial_train_data = np.concatenate((
                initial_train_data, train_data[select_index]), axis=0)
            initial_train_label = np.concatenate(
                (initial_train_label, train_label[select_index]), axis=0)
        # delete the selected data from client_dataset
        train_data = np.delete(train_data, select_index, axis=0)
        train_label = np.delete(train_label, select_index, axis=0)
    return train_data, train_label, initial_train_data, initial_train_label


def main():
    # the path of sorted data(by label) csv files
    sort_train_data_path = os.path.join(os.path.dirname(
        __file__), '../../ISOLET/sorted_data_csv/train_data.csv')
    sort_train_label_path = os.path.join(os.path.dirname(
        __file__), '../../ISOLET/sorted_data_csv/train_label.csv')
    sort_test_data_path = os.path.join(os.path.dirname(
        __file__), '../../ISOLET/sorted_data_csv/test_data.csv')
    sort_test_label_path = os.path.join(os.path.dirname(
        __file__), '../../ISOLET/sorted_data_csv/test_label.csv')

    # load the csv file and transfer them into numpy array
    sort_train_data = pd.read_csv(sort_train_data_path)
    sort_train_data = np.asarray(sort_train_data.iloc[:, 1:])

    sort_train_label = pd.read_csv(sort_train_label_path)
    sort_train_label = np.asarray(sort_train_label.iloc[:, 1:])

    sort_test_data = pd.read_csv(sort_test_data_path)
    sort_test_data = np.asarray(sort_test_data.iloc[:, 1:])

    sort_test_label = pd.read_csv(sort_test_label_path)
    sort_test_label = np.asarray(sort_test_label.iloc[:, 1:])

    # check the size of each dataset
    print("Size of training data:{}".format(sort_train_data.shape))
    print("Size of training label:{}".format(sort_train_label.shape))
    print("Size of testing data:{}".format(sort_test_data.shape))
    print("Size of testing label:{}".format(sort_test_label.shape))

    def init_IM_vector(nof_dimension, nof_feature):
        ''' 創建feature數量個vector element 每個element為bipolar(1,-1)'''
        IM_vector = {}
        for i in range(1, nof_feature+1):
            # np.random.choice([1,-1],size) 隨機二選一 size代表選幾次
            IM_vector['feature'+str(i)] = np.random.choice(
                [1, -1], nof_dimension).reshape(1, nof_dimension).astype(int)
        return IM_vector

    def init_CIM_vector(nof_dimension, level):
        ''' slice continuous signal into self.level parts'''
        # 每往上一個self.level就改 D/2/(self.level-1)個bit

        CIM_vector = {}
        nof_change = nof_dimension//(2*(level-1))
        if nof_dimension/2/(level-1) != floor(nof_dimension/2/(level-1)):
            print("warning! D/2/(level-1) is not an integer,", end=' ')
            print(
                "change the dim so that the maximum CIM vector can be orthogonal to the minimum CIM vector")

        CIM_vector[0] = np.random.choice(
            [1, -1], nof_dimension).reshape(1, nof_dimension).astype(int)

        for lev in range(1, level):
            # 每個level要改D/2/(level-1)個bit 並且從 D/2/(level-1) * (lev-1)開始改
            # 這裡用到的觀念叫做deep copy 非常重要
            # 只copy value而不是像python assign一樣是share 物件
            CIM_vector[lev] = CIM_vector[lev-1].copy()
            for index in range(nof_change * (lev-1), nof_change * (lev)):

                CIM_vector[lev][0][index] *= -1
        return CIM_vector

    parser = argparse.ArgumentParser()
    parser.add_argument('-K', help="number of clients")
    parser.add_argument('-D', help='Dimension of hypervectors')
    args = parser.parse_args()
    '''
    * Parition the dataset into initial data and client dataset
    '''
    # @sort_XXX: dataset for clients
    # @initial_XXX: dataset for initialization
    sort_train_data, sort_train_label, initial_train_data, initial_train_label = partition_initial_client(
        sort_train_data, sort_train_label)
    '''
    * Parameter setup
    * @K: number of clients
    * @D: Dimension of hypervectors
    * @l: level of CIM vectors
    * @Nof_feature: # of feature for each sample
    * @PCA_Projection: whether use PCA or not
    * @L: # of labels
    * @CIM : Communal CIM vectors
    * @IM : Communal IM vectors
    * @BinaryAM : Predict with Binary AM or IntegerAM
    '''
    Base_model = {}
    if args.K:
        Base_model['K'] = int(args.K)
    else:
        Base_model['K'] = 20
    if args.D:
        Base_model['D'] = int(args.D)
    else:
        Base_model['D'] = 1000
    Base_model['l'] = 21
    Base_model['Nof_feature'] = sort_train_data.shape[1]
    Base_model['PCA_Projection'] = False
    Base_model['BinaryAM'] = True
    Base_model['L'] = 26
    Base_model['CIM'] = init_CIM_vector(Base_model['D'], Base_model['l'])
    Base_model['IM'] = init_IM_vector(
        Base_model['D'], Base_model['Nof_feature'])
    # use initial dataset to initialize the maximum and minimum of base model
    Base_model['maximum'] = np.max(initial_train_data, axis=0)
    Base_model['minimum'] = np.min(initial_train_data, axis=0)
    # save the setup as pickle file
    with open(os.path.join(os.path.dirname(__file__), '../Setup.pickle'), 'wb') as f:
        pickle.dump(Base_model, f)
    '''Save initial dataset as csv file'''
    initial_train_data_df = pd.DataFrame(initial_train_data, columns=[
                                         'feature'+str(i) for i in range(1, len(initial_train_data[0])+1)])
    initial_train_data_df.to_csv(os.path.join(
        os.path.dirname(__file__), 'initial_train_data.csv'))

    initial_train_label_df = pd.DataFrame(initial_train_label, columns=[
        'Label'])
    initial_train_label_df.to_csv(os.path.join(
        os.path.dirname(__file__), 'initial_train_label.csv'))

    '''
    * Partition client data into 2K shards.
    * Each client will have 2 shards of data-> Non IID balanced
    '''
    ''' train_data_part'''
    # @nof_shards: how many shards of data should the total dataset be partitioned to
    # @size_of_shard: size of data in each shards
    nof_shards = 2*Base_model['K']
    if sort_train_data.shape[0]/nof_shards != int(sort_train_data.shape[0]/nof_shards):
        print("Unbalanced Train dataset! Please Modify K(# of clients) if balanced dataset is demanded.")
    size_of_shard = int(sort_train_data.shape[0]//nof_shards)
    shard_index = np.array([i for i in range(nof_shards)], dtype=int)

    for client in range(1, Base_model['K']+1):
        # For each client, random sample two shards of data from the total dataset
        # That is, randomly select two index and delete them from the index list.
        # data_index: selected index for client
        # shard_index: total index list
        data_index = [-1, -1]
        data_index[0] = int(np.random.choice(shard_index, size=1))
        shard_index = np.delete(shard_index, np.where(
            shard_index == data_index[0]))
        data_index[1] = int(np.random.choice(shard_index, size=1))
        shard_index = np.delete(shard_index, np.where(
            shard_index == data_index[1]))

        ''' Partition the data and save them as csv file in each client directory'''
        if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'client'+str(client))):
            os.makedirs(os.path.join(os.path.dirname(
                os.path.dirname(__file__)), 'client'+str(client)))
        # the range of client_data lies between data_index*size_of shard and (data_index+1) * (size of shard)
        client_data = sort_train_data[data_index[0] *
                                      size_of_shard: (data_index[0]+1)*size_of_shard]
        client_label = sort_train_label[data_index[0] *
                                        size_of_shard: (data_index[0]+1)*size_of_shard]
        client_data = np.concatenate((client_data, sort_train_data[data_index[1] *
                                                                   size_of_shard: (data_index[1] + 1)*size_of_shard]), axis=0)
        client_label = np.concatenate((client_label, sort_train_label[data_index[1] *
                                                                      size_of_shard: (data_index[1] + 1)*size_of_shard]), axis=0)
        # save the data as csv files
        print("Save training csv files for client"+str(client)+"...", end=' ')
        client_data_df = pd.DataFrame(
            client_data, columns=['feature'+str(i) for i in range(1, Base_model['Nof_feature'] + 1)])
        client_data_df.to_csv(os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'client'+str(client), 'train_data.csv'))
        client_label_df = pd.DataFrame(client_label, columns=['Label'])
        client_label_df.to_csv(os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'client'+str(client), 'train_label.csv'))
        print("Complete!")


if __name__ == "__main__":
    main()
