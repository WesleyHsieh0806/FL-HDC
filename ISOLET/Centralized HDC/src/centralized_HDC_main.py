import random
import os
import time
import numpy as np
import pandas as pd
import HDC_Centralized as HDC
np.random.seed(0)
'''
Author: Wesley Hsieh
This file is for experiments of Centralized HDC on EISOLET Dataset
Experiments includes:
1.Centralized HDC on Total IID ISOLET Dataset
2.Cetralized HDC+Retrain on Total IID ISOLET Dataset

'''


def main():
    file_dir = os.path.dirname(__file__)
    train_data_path = os.path.join(
        file_dir, "../../ISOLET/data_csv/train_data.csv")
    train_label_path = os.path.join(
        file_dir, "../../ISOLET/data_csv/train_label.csv")
    test_data_path = os.path.join(
        file_dir, "../../ISOLET/data_csv/test_data.csv")
    test_label_path = os.path.join(
        file_dir, "../../ISOLET/data_csv/test_label.csv")
    # Read training data
    train_data = pd.read_csv(train_data_path)
    x = np.asarray(train_data, dtype=np.float)

    # Read training label
    train_label = pd.read_csv(train_label_path)
    y = np.asarray(train_label.iloc[:, 1:], dtype=np.int)

    # Read testing data
    test_data = pd.read_csv(test_data_path)
    test_x = np.asarray(test_data.iloc[:, 1:], dtype=np.float)

    # Read testing label
    test_label = pd.read_csv(test_label_path)
    test_y = np.asarray(test_label.iloc[:, 1:], dtype=np.int)

    print("Size of training data:{}".format(x.shape))
    print("Size of training label:{}".format(y.shape))
    print("Size of testing data:{}".format(test_x.shape))
    print("Size of testing label:{}".format(test_y.shape))

    if not os.path.isdir(os.path.join(
            file_dir, '../Result/binary_train')):
        os.makedirs(os.path.join(
            file_dir, '../Result/binary_train'))
    """
        HDC Training Part
    """
    '''# Parameter Setup'''
    # Dimension = [1000*i for i in range(1, 11)]
    Dimension = [1000, 2000, 5000, 10000]
    n_of_class = 26
    level = 21
    n_of_feature = len(x[0])
    # the result of each parameter setup is the average of 5 times
    average_time = 10
    result = {}
    Time = {}
    for dimension in Dimension:
        # Initialize the dictionary to record accuracy and training time
        result['dim'+str(dimension)] = 0.
        Time[str(dimension)+' time'] = 0.
        for i in range(average_time):
            # Initialize HDC Model
            ISOLET = HDC.HDC(dimension, n_of_class, n_of_feature,
                             level=level, PCA_projection=False)

            # Begin training
            start = time.time()
            ISOLET.train(x[:], y[:])

            # Record the training time
            train_time = time.time()-start

            # Start Testing
            start = time.time()
            y_pred = ISOLET.test(test_x[:])
            test_time = time.time()-start
            acc = ISOLET.accuracy(y_true=test_y[:], y_pred=y_pred[:])
            print('Training time:{:.3f} Testing time:{:.3f} Dimension:{} Level:{}'.format(
                train_time, test_time, dimension, level))
            print("Accuracy:{:.4f}".format(acc))

            # Record accuracy and training time
            result['dim'+str(dimension)] += acc
            Time[str(dimension)+' time'] += train_time
        # Average the accuracy and training time
        result['dim'+str(dimension)] /= average_time
        Time[str(dimension)+' time'] /= average_time

        Acc = pd.DataFrame([result['dim'+str(dimension)]], columns=[dimension])
        Acc.to_csv(os.path.join(
            file_dir, '../Result/train_8000+4000/Accuracy' +
                   str(dimension)+'.csv'))
        TTime = pd.DataFrame([Time[str(dimension)+' time']],
                             index=[0], columns=[dimension])
        TTime.to_csv(os.path.join(
            file_dir, '../Result/train_8000+4000/Training_time' +
                     str(dimension)+'.csv'))

    # Save the accuracy and training time into csv file
    result = pd.DataFrame(result, index=[0])
    result.to_csv(os.path.join(
        file_dir, '../Result/binary_train/Accuracy.csv'))
    Time = pd.DataFrame(Time, index=[0])
    Time.to_csv(os.path.join(
        file_dir, '../Result/binary_train/Training_time.csv'))


if __name__ == "__main__":
    main()
