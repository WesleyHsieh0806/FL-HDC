import numpy as np
import pandas as pd
import os

'''
*   Sort the data by labels and save them as csv files(for Non-IID case)
'''
file_dir = os.path.dirname(__file__)
if not os.path.isdir(os.path.join(file_dir, '../UCIHAR/sorted_data_csv')):
    os.makedirs(os.path.join(file_dir, '../UCIHAR/sorted_data_csv'))


def load_data(train_path, test_path):
    '''
    Input: Path of train and test datasets csv files
    Return: train_data, train_label, test_data, test_label
    Since The labels and data are both in one csv file, Here we will partition them into data and label parts
    '''
    # Load the train and test datasets into numpy array
    with open(train_path, 'r', encoding='utf-8-sig')as f:
        train_dataset = [line.strip().split(',') for line in f]
    train_dataset = np.array(train_dataset).astype(np.float64)
    with open(test_path, 'r', encoding='utf-8-sig')as f:
        test_dataset = [line.strip().split(',') for line in f]
    test_dataset = np.array(test_dataset).astype(np.float64)

    # Partition data and label from dataset
    train_data = train_dataset[:, :-1]
    train_label = train_dataset[:, -1]
    test_data = test_dataset[:, :-1]
    test_label = test_dataset[:, -1]
    # Change the label to int to prevent imprecision
    train_label = train_label.astype(np.int32, copy=False)
    test_label = test_label.astype(np.int32, copy=False)

    # Check the shape of the data and label
    print("Size of training data:{}".format(train_data.shape))
    print("Size of training label:{}".format(train_label.shape))
    print("Size of testing data:{}".format(test_data.shape))
    print("Size of testing label:{}".format(test_label.shape))
    return train_data, train_label, test_data, test_label


# the path of each csv file
train_path = os.path.join(file_dir, '../UCIHAR/data_csv/train.csv')
test_path = os.path.join(file_dir, '../UCIHAR/data_csv/test.csv')

# load the data and label from given path
train_data, train_label, test_data, test_label = load_data(
    train_path, test_path)
# the path to be saved
train_data_path = os.path.join(
    file_dir, '../UCIHAR/data_csv/train_data.csv')
train_label_path = os.path.join(
    file_dir, '../UCIHAR/data_csv/train_label.csv')
test_data_path = os.path.join(
    file_dir, '../UCIHAR/data_csv/test_data.csv')
test_label_path = os.path.join(
    file_dir, '../UCIHAR/data_csv/test_label.csv')

# Partition the data and label and save them as csv files
train_data_df = pd.DataFrame(train_data, columns=[
    'Feature'+str(i) for i in range(1, len(test_data[0])+1)])
train_data_df.to_csv(train_data_path)

train_label_df = pd.DataFrame(train_label, columns=['Label'])
train_label_df.to_csv(train_label_path)

test_data_df = pd.DataFrame(test_data, columns=[
    'Feature'+str(i) for i in range(1, len(test_data[0])+1)])
test_data_df.to_csv(test_data_path)

test_label_df = pd.DataFrame(test_label, columns=['Label'])
test_label_df.to_csv(test_label_path)

# the path to be saved
sort_train_data_path = os.path.join(
    file_dir, '../UCIHAR/sorted_data_csv/train_data.csv')
sort_train_label_path = os.path.join(
    file_dir, '../UCIHAR/sorted_data_csv/train_label.csv')
sort_test_data_path = os.path.join(
    file_dir, '../UCIHAR/sorted_data_csv/test_data.csv')
sort_test_label_path = os.path.join(
    file_dir, '../UCIHAR/sorted_data_csv/test_label.csv')

# sort the data by label
train_index = train_label.argsort(kind='mergesort', axis=0)
sort_train_data = train_data[train_index]
sort_train_label = train_label[train_index]
test_index = test_label.argsort(kind='mergesort', axis=0)
sort_test_data = test_data[test_index]
sort_test_label = test_label[test_index]

# Save them as csv files
sort_train_data = pd.DataFrame(sort_train_data, columns=[
                               'Feature'+str(i) for i in range(1, len(test_data[0])+1)])
sort_train_data.to_csv(sort_train_data_path)

sort_train_label = pd.DataFrame(sort_train_label, columns=['Label'])
sort_train_label.to_csv(sort_train_label_path)

sort_test_data = pd.DataFrame(sort_test_data, columns=[
                              'Feature'+str(i) for i in range(1, len(test_data[0])+1)])
sort_test_data.to_csv(sort_test_data_path)

sort_test_label = pd.DataFrame(sort_test_label, columns=['Label'])
sort_test_label.to_csv(sort_test_label_path)

# Print the number of data for each label
for i in range(1, 7):
    print("Size of class {}:{} in train data".format(
        i, sort_train_label[sort_train_label == i].count(axis=0).values))
for i in range(1, 7):
    print("Size of class {}:{} in test data".format(
        i, sort_test_label[sort_test_label == i].count(axis=0).values))
