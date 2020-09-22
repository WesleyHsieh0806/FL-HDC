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
    train_data_path = os.path.join(os.path.dirname(
        __file__), "../../ISOLET/data_csv/train_data.csv")
    train_label_path = os.path.join(os.path.dirname(
        __file__), "../../ISOLET/data_csv/train_label.csv")
    test_data_path = os.path.join(os.path.dirname(
        __file__),  "../../ISOLET/data_csv/test_data.csv")
    test_label_path = os.path.join(os.path.dirname(
        __file__), "../../ISOLET/data_csv/test_label.csv")
    # Read training data
    train_data = pd.read_csv(train_data_path)
    x = np.asarray(train_data.iloc[:, 1:], dtype=np.float)

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

    if not os.path.isdir(os.path.join(os.path.dirname(
            __file__), '../Result/binary_retrain_60000+10000')):
        os.makedirs(os.path.join(os.path.dirname(
            __file__), '../Result/binary_retrain_60000+10000'))
    """
        HDC Training Part
    """
    '''# Parameter Setup'''
    # Dimension = [1000*i for i in range(1, 11)]
    Dimension = [10000]
    n_of_class = 26
    level = 21
    n_of_feature = len(x[0])
    # the result of each parameter setup is the average of 5 times
    average_time = 1
    result = {}
    Time = {}
    for dimension in Dimension:
        # Initialize the dictionary to record accuracy and training time
        result['dim'+str(dimension)] = []
        Time[str(dimension)+'time'] = []
    for i in range(average_time):
        for dimension in Dimension:
            # Initialize HDC Model-BinaryAM
            ISOLET = HDC.HDC(dimension, n_of_class, n_of_feature,
                             level=level, PCA_projection=False, binaryAM=True)

            # Begin training
            start = time.time()
            ISOLET.train(x[:], y[:])
            train_time = time.time()-start

            # Begin testing(One-shot)
            start = time.time()
            y_pred = ISOLET.test(test_x[:])
            test_time = time.time()-start
            acc = ISOLET.accuracy(y_true=test_y[:], y_pred=y_pred[:])
            print('Training time:{:.3f} Testing time:{:.3f} Dimension:{} Level:{}'.format(
                train_time, test_time, dimension, level))
            print("Accuracy:{:.4f}".format(acc))

            # # Retrain the saved model
            # ISOLET.load_model('HDC_model.pickle')
            # Retrain
            print("{:=^40}".format("Start Retraining"))
            _, acc_history, time_history = ISOLET.retrain(
                test_x[:], test_y[:], x[:], y[:], num_epoch=30, train_acc_demand=0.85, batch_size=len(x)//3, save_path='HDC_model.pickle')

            # Record accuracy and training time
            result['dim'+str(dimension)].append(acc_history)
            Time[str(dimension)+'time'].append(time_history)

            # Save the accuracy_history into csv files
            Acc = pd.DataFrame(np.array(result['dim'+str(dimension)]),
                               columns=[i+1 for i in range(len(result['dim'+str(dimension)][0]))])
            Acc.to_csv(os.path.join(os.path.dirname(
                __file__), '../Result/binary_retrain_60000+10000/Accuracy' +
                str(dimension)+'.csv'))
            # Save retrain execution time into csv files
            TTime = pd.DataFrame(np.array(Time[str(dimension)+'time']),
                                 columns=[i+1 for i in range(len(result['dim'+str(dimension)][0]))])
            TTime.to_csv(os.path.join(os.path.dirname(
                __file__), '../Result/binary_retrain_60000+10000/retraining_time' +
                str(dimension)+'.csv'))


if __name__ == "__main__":
    main()
