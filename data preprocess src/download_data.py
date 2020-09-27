import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
"""
Download Mnist Training data and Testing data
"""
# Construct a tf.data.Dataset
train, test = tfds.as_numpy(tfds.load('mnist', split=['train', 'test'],
                                      data_dir='../MNIST/', batch_size=-1))
print("Size of training Image:{}".format(train['image'].shape))
print("Size of training Image:{}".format(test['image'].shape))
'''
Save them as csv file (which is convenient to HDC)
'''
if not os.path.isdir('../MNIST/data_csv/'):
    os.makedirs('../MNIST/data_csv/')
# flatten the image
train_data = train['image'].reshape([len(train['image']), -1])
train_label = train['label'].reshape(len(train['label']))
test_data = test['image'].reshape([len(test['image']), -1])
test_label = test['label'].reshape(len(test['label']))
if not os.path.isfile('../MNIST/data_csv/train_data.csv'):
    train_data = pd.DataFrame(
        train_data, columns=['Feature'+str(i) for i in range(1, len(train_data[0])+1)])
    train_data.to_csv('../MNIST/data_csv/train_data.csv')
# Create csv file for training label
if not os.path.isfile('../MNIST/data_csv/train_label.csv'):
    train_label = pd.DataFrame(train_label, columns=['Label'])
    train_label.to_csv('../MNIST/data_csv/train_label.csv')

# Create csv file for testing data

if not os.path.isfile('../MNIST/data_csv/test_data.csv'):
    test_data = pd.DataFrame(
        test_data, columns=['Feature'+str(i) for i in range(1, len(test_data[0])+1)])
    test_data.to_csv('../MNIST/data_csv/test_data.csv')

# Create csv file for testing label
if not os.path.isfile('../MNIST/data_csv/test_label.csv'):
    test_label = pd.DataFrame(test_label, columns=['label'])
    test_label.to_csv('../MNIST/data_csv/test_label.csv')
