import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# the path of csv file
train_accuracy_path = './train_8000+4000/Accuracy.csv'
trainPCA_accuracy_path = './trainPCA_8000+4000/Accuracy.csv'
retrain_accuracy_path = './Retrain_8000+4000/Accuracy.csv'
retrainPCA_accuracy_path = './retrainPCA_8000+4000/Accuracy.csv'

# transfer the results into array
train_accuracy = pd.read_csv(train_accuracy_path)
train_accuracy = train_accuracy.iloc[0, 1:]

trainPCA_accuracy = pd.read_csv(trainPCA_accuracy_path)
trainPCA_accuracy = trainPCA_accuracy.iloc[0, 1:]

retrain_accuracy = pd.read_csv(retrain_accuracy_path)
retrain_accuracy = retrain_accuracy.iloc[0, 1:]

retrainPCA_accuracy = pd.read_csv(retrainPCA_accuracy_path)
retrainPCA_accuracy = retrainPCA_accuracy.iloc[0, 1:]

# plot the result
x = [1000, 2000, 5000, 10000]

plt.plot(x, train_accuracy, 'bo-', label='One-shot')
plt.plot(x, trainPCA_accuracy, color='b', marker='o',
         linestyle='--', label='One-shot with PCA')
plt.plot(x, retrain_accuracy, 'ro-', label='retrain')
plt.plot(x, retrainPCA_accuracy, color='r', marker='o',
         linestyle='--', label='retrain with PCA')
plt.legend()
plt.xticks([1000*i for i in range(1, 11)])
plt.grid()
plt.savefig('./Accuracy.png')
plt.close()
