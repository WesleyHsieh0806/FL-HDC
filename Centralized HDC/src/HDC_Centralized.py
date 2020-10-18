# -*- coding=utf-8 -*-
import pickle
import time
import numpy as np
import pandas as pd
from math import floor
import sys
from multiprocessing import Pool
import multiprocessing as mul
from sklearn.decomposition import PCA
import random
import json
import os
from math import ceil
# you can use HDC.help() to realize what parameters are necessary to be filled in
# self回傳object資訊 沒有self回傳class資訊
# 2020/6/7:Update PCA functions
# 2020/7/29:Update Retrain_earlystop, Add integer AM(Only binary AM before)-> binary AM for testing, integer AM for retrain
# 2020/8/7:Update Retrain
np.random.seed(0)
random.seed(0)


class HDC:

    # attritbue :
    # nof_feature
    # nof_class
    # nof_dimension
    # level
    # IM_vector
    # CIM_vector
    # Prototype_vector
    # self.maximum = np.max(x, axis=0) train data 每個 feature的最大值
    # self.minimum = np.min(x, axis=0)
    # self.difference = self.maximum - self.minimum
    # self.PCA_projection
    # self.binaryAM (use binaryAM or IntegerAM)
    def __init__(self, dim=10000, nof_class=0, nof_feature=0, level=21, PCA_projection=False, binaryAM=True):
        ''' initialize some necessary attribute and data'''
        # @nof_feature:feature數量(how many IM vector?) -->not necessary
        # @dim:vector dimension
        # @nof_class:class數量(how many prototype vector)
        self.level = int(level)
        self.nof_feature = int(nof_feature)
        self.nof_dimension = int(dim)
        self.nof_class = int(nof_class)
        # determine whether use PCA to project the features or not
        self.PCA_projection = PCA_projection
        # binaryAM or IntegerAM
        self.binaryAM = binaryAM

    def train(self, x, y, IM_vector=None, CIM_vector=None, maximum=None, minimum=None, difference=None):
        ''' use train data x y to train prototype vector'''
        # x,y須為ndarray
        # 此步驟 需要創建IM(字典) CIM(字典) Prototype vector(字典)
        if self.PCA_projection:
            x = self.PCA(x)
        self.nof_feature = len(x[0])
        self.init_IM_vector(IM_vector)
        self.init_CIM_vector(CIM_vector)
        self.init_prototype_vector()
        spatial_vector = np.zeros(
            (len(x), 1, self.nof_dimension)).astype(int)
        # 因為要將x每個feature的value根據數值切成level個等級 所以要記住某個範圍的數值
        # 以level=21為例子 假如數值範圍是0~20 就是0是一個level
        # 但這裡實作 我打算將(0~20)/21 當作level 0
        # 要將maximum這些存下來 才能用在test
        if maximum is not None:
            # test if the maximum should be initialized with demanded values(FL settings)
            print('Initial Maximum with given values')
            self.maximum = maximum
        else:
            self.maximum = np.max(x, axis=0)
        if minimum is not None:
            # test if the minimum should be initialized with demanded values(FL settings)
            print('Initial Minimum with given values')
            self.minimum = minimum
        else:
            self.minimum = np.min(x, axis=0)
        self.difference = self.maximum - self.minimum + 1e-8

        self.encoder_spatial_vector(x, spatial_vector, y)

        for CLASS in range(0, self.nof_class):
            # 這裡是train的最後了 把prototype vector變回1 -1-->形成'binary AM'
            self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS]
                                                   > 0] = 1
            self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS]
                                                   < 0] = -1

    def PCA(self, x):
        print(
            "Project {}-dim features to dimension".format(len(x[0])), end=' ')
        self.pca = PCA(n_components=20)
        self.pca.fit(x)
        x = self.pca.fit_transform(x)
        print("{}".format(len(x[0])))
        return x

    def PCA_test(self, x):
        x = self.pca.transform(x)
        return x

    def retrain_earlystop(self, test_x, test_y, train_x, train_y, train_acc_demand=0.85, batch_size=400, save_path='HDC_model.pickle'):
        ''' 
        Retrain the prototype vector(integer AM) with training data and stops when 
        the test-set accuracy drops 3times continuously and the train_acc acheives the demanded level
        @train_acc_demand: the least accuracy which should be achieved before stop retraining
        @batch_size : The number of data passed through between two prototype AM update
        @save_path : the path where to save the model as pickle file
        Return : best test accuracy achieved in retraining phase
        '''
        # train_x:(n,feature) train_y(n,1)
        best_acc = 0.
        last_acc = -1.
        train_acc = 0.
        acc = 0.
        retrain_epoch = 1
        drop_time = 0
        total_batch = ceil(len(train_x)/batch_size)
        if self.PCA_projection:
            # in retrain phase, we have to project the training data first if PCA was used in training
            # testing data do not need to be projected here since self.test already do it for us
            train_x = self.PCA_test(train_x)
        while (drop_time < 3) or (train_acc < train_acc_demand):
            '''If the test-set accuracy gets lower, then retrain stops'''
            same = 0
            for batch in range(total_batch):
                # batch retraining: Update binary AM and test-set accuracy for each batch
                for data in range(batch_size):
                    data_index = batch*batch_size + data
                    if data_index < len(train_x):
                        # there are still data which have not been passed encoded
                        query_vector = np.zeros(
                            (1, self.nof_dimension)).astype(int)
                        print("-- Retrain Epoch{} --[{}/{}] Dimension:{} Level:{}".format(retrain_epoch, data_index+1,
                                                                                          len(train_x), self.nof_dimension, self.level), end='\r')
                        # Prediction
                        predicted_class, query_vector = self.encoder_query_vector(
                            train_x[data_index], query_vector, data_index, retrain=True)
                        # if the data is wrongly predicted, subtract the mismatched prototype vector by query_vector
                        if predicted_class != train_y[data_index][0]:
                            # In this case : predicted_class = mismatched class
                            # the real class is train_y[data]
                            self.Prototype_vector['integer'][predicted_class] -= query_vector
                            self.Prototype_vector['integer'][train_y[data_index][0]
                                                             ] += query_vector
                        else:
                            same += 1
                    else:
                        # All data have been encoded
                        break
                print("\nUpdate Test-set accuracy...")
                ''' Binarize Prototype Vector and update binary AM'''
                for CLASS in range(0, self.nof_class):
                    # After Retraining, the binary Prototype vector should be updated()
                    # As a result, we have to binarize them (>0 --> 1   <0 --> -1)
                    # Special case: if an element is 0, then randomly change it into 1 or -1
                    self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS]
                                                           > 0] = 1
                    self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS]
                                                           < 0] = -1
                    '''
                    * In NonIID setting, the zero elements should not be changed to 1 -1, since there are many classes
                    * (which do not exist in the local dataset) vectors which have only zero elements
                    '''
                    # self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS] == 0] = np.random.choice(
                    #     [1, -1], size=np.count_nonzero(self.Prototype_vector['integer'][CLASS] == 0))
                ''' Print the training_accuracy for each batch'''
                if self.PCA_projection:
                    # We have to turn off the PCA_projection since train_x has already been projected
                    self.PCA_projection = False
                    train_y_pred = self.test(train_x)
                    self.PCA_projection = True
                else:
                    train_y_pred = self.test(train_x)
                train_acc = self.accuracy(y_true=train_y, y_pred=train_y_pred)
                print("Training accuracy:{:.4f}".format(train_acc))

                '''Predict the test data'''
                y_pred = self.test(test_x)
                ''' Acquire the test-accuracy to see the results of retraining'''
                # Update accuracy
                last_acc = acc
                acc = self.accuracy(y_true=test_y, y_pred=y_pred)
                # Update drop time
                if acc < last_acc:
                    drop_time += 1
                else:
                    drop_time = 0
                if acc > best_acc:
                    best_acc = acc
                    print("Currently best accuracy:{:.4f}".format(best_acc))
                    self.save_model(path=save_path)
                print("Test accuracy:{:.4f}".format(acc))
                print("Last accuracy:{:.4f}".format(
                    last_acc))
                # Test if the retrain should be continued
                if not((drop_time < 3) or (train_acc < train_acc_demand)):
                    break
            retrain_epoch += 1

        print("\nRetrain Complete! Best accuracy:{:.4f}".format(best_acc))
        return best_acc

    def retrain(self, test_x, test_y, train_x, train_y, num_epoch=5, train_acc_demand=0.85, batch_size=400, save_path='HDC_model.pickle'):
        ''' 
        Retrain the prototype vector(integer AM) with training data and stops after num_epoch
        @train_acc_demand: the least accuracy which should be achieved before stop retraining
        @batch_size : The number of data passed through between two prototype AM update
        @save_path : the path where to save the model as pickle file
        Return : best test accuracy achieved in retraining phase
        '''
        # train_x:(n,feature) train_y(n,1)
        best_acc = 0.
        last_acc = -1.
        train_acc = 0.
        acc = 0.
        acc_history = []
        time_history = []
        total_batch = ceil(len(train_x)/batch_size)
        if self.PCA_projection:
            # in retrain phase, we have to project the training data first if PCA was used in training
            # testing data do not need to be projected here since self.test already do it for us
            train_x = self.PCA_test(train_x)
        for retrain_epoch in range(1, num_epoch+1):
            # Record the execution time of each epoch
            execution_time = 0.
            '''If the test-set accuracy gets lower, then retrain stops'''
            for batch in range(total_batch):
                # Record the start time of each iteration
                batch_start = time.time()
                # batch retraining: Update binary AM and test-set accuracy for each batch
                for data in range(batch_size):
                    data_index = batch*batch_size + data
                    if data_index < len(train_x):
                        # there are still data which have not been passed encoded
                        query_vector = np.zeros(
                            (1, self.nof_dimension)).astype(int)
                        print("-- Retrain Epoch{} --[{}/{}] Dimension:{} Level:{}".format(retrain_epoch, data_index+1,
                                                                                          len(train_x), self.nof_dimension, self.level), end='\r')
                        # Prediction
                        predicted_class, query_vector = self.encoder_query_vector(
                            train_x[data_index], query_vector, data_index, retrain=True)
                        # if the data is wrongly predicted, subtract the mismatched prototype vector by query_vector
                        if predicted_class != train_y[data_index][0]:
                            # In this case : predicted_class = mismatched class
                            # the real class is train_y[data]
                            self.Prototype_vector['integer'][predicted_class] -= query_vector
                            self.Prototype_vector['integer'][train_y[data_index][0]
                                                             ] += query_vector

                    else:
                        # All data have been encoded
                        break
                # Record the time of the end of each batch
                execution_time += time.time()-batch_start
                print("\nUpdate Test-set accuracy...")
                ''' Binarize Prototype Vector and update binary AM'''
                for CLASS in range(0, self.nof_class):
                    # After Retraining, the binary Prototype vector should be updated()
                    # As a result, we have to binarize them (>0 --> 1   <0 --> -1)
                    # Special case: if an element is 0, then randomly change it into 1 or -1
                    self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS]
                                                           > 0] = 1
                    self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS]
                                                           < 0] = -1
                    '''
                    * In NonIID setting, the zero elements should not be changed to 1 -1, since there are many classes
                    * (which do not exist in the local dataset) vectors which have only zero elements
                    '''
                    # self.Prototype_vector['binary'][CLASS][self.Prototype_vector['integer'][CLASS] == 0] = np.random.choice(
                    #     [1, -1], size=np.count_nonzero(self.Prototype_vector['integer'][CLASS] == 0))
                ''' Print the training_accuracy for each batch'''
                if self.PCA_projection:
                    # We have to turn off the PCA_projection since train_x has already been projected
                    self.PCA_projection = False
                    train_y_pred = self.test(train_x)
                    self.PCA_projection = True
                else:
                    train_y_pred = self.test(train_x)
                # train_acc = self.accuracy(y_true=train_y, y_pred=train_y_pred)
                # print("Training accuracy:{:.4f}".format(train_acc))

                '''Predict the test data'''
                y_pred = self.test(test_x)
                ''' Acquire the test-accuracy to see the results of retraining'''
                # Update accuracy
                last_acc = acc
                acc = self.accuracy(y_true=test_y, y_pred=y_pred)
                # Update drop time
                if acc > best_acc:
                    best_acc = acc
                    print("Currently best accuracy:{:.4f}".format(best_acc))
                    self.save_model(path=save_path)
                print("Test accuracy:{:.4f}".format(acc))
                print("Last accuracy:{:.4f}".format(
                    last_acc))
                # Record the accuracy of each iteration in acc_history
                acc_history.append(acc)
                # Record the execution of each iteration in time_history
                time_history.append(execution_time)
        print("\nRetrain Complete! Best accuracy:{:.4f}".format(best_acc))
        return best_acc, acc_history, time_history

    def test(self, test_x):
        ''' return the predicted y array(class) '''
        # 首先要將test data經過同樣encoder 並產生query vector
        if self.PCA_projection:
            test_x = self.PCA_test(test_x)
        query_vector = np.zeros(
            (len(test_x), 1, self.nof_dimension)).astype(int)
        self.y_pred = np.zeros((len(test_x), 1))

        ''' encoding and prediction'''
        # Operate Multiprocess with pool(from multiprocessing)(多個CPU核心去做運算)
        # Pool() 代表利用cpu最大核心數量去跑
        # 用 starmap function他就會自動分配資料讓核心去跑 並回傳每次function的結果成一個list
        # 這裡要注意用多核心去跑的時候 即使在function裡面改了self.pred的value 也不會改動到self.pred的value
        # 可以在裡面改self.y_pred[0][0] 並在這裡print看看就可知道
        start = time.time()
        pool = Pool()

        self.y_pred = np.array([pool.starmap_async(self.encoder_query_vector, [
            (test_x[data, :], query_vector[data, :], data) for data in range(len(test_x))]).get()]).reshape((len(test_x), 1))
        pool.close()
        pool.join()
        end = time.time()

        return self.y_pred

    def result_to_csv(self, output='./result.csv'):
        '''output the result of prediction as csv file'''
        with open(output, 'w') as f:
            f.write('data,class\n')
            for data in range(len(self.y_pred)):
                f.write('{},{}\n'.format(str(data), self.y_pred[data][0]))

    def accuracy(self, y_true, y_pred=None):
        '''return the accuracy of the prediction'''
        same = 0
        if y_pred is None:
            # if y_pred is not given, use self.y_pred we obtain in test function
            if self.y_pred:
                y_pred = self.y_pred

        for data in range(len(y_true)):
            if y_pred[data, 0] == y_true[data, 0]:
                same += 1
        return same / len(y_pred)

    def cosine_similarity(self, Query_vector, Prototpye_vector):
        '''return cos(A,B)=|A'*B'|=|C| C is the sum of element'''
        # 這個function只處理1對1的cosine similarity
        cos_sim = np.dot(Query_vector, Prototpye_vector.T) / \
            (np.linalg.norm(Query_vector)*np.linalg.norm(Prototpye_vector)+1e-9)
        return cos_sim

    def most_similar_class(self, Query_vector):
        '''return the number of class(0~self.nof_class-1)which is the most similar to query_vector'''
        maximum = -100
        max_class = -1
        for Class in range(0, self.nof_class):
            # Compare similarity with AM
            if self.binaryAM:
                similarity = self.cosine_similarity(
                    Query_vector, self.Prototype_vector['binary'][Class])
            else:
                similarity = self.cosine_similarity(
                    Query_vector, self.Prototype_vector['integer'][Class])
            if similarity > maximum:
                maximum = similarity
                max_class = Class

        return max_class

    def init_IM_vector(self, IM_vector=None):
        if IM_vector:
            # test if the IM vector should be initialized with demanded vectors
            print('Initial IM with given vectors')
            self.IM_vector = IM_vector
            return
        ''' Construct #offeature vectors with each element being bipolar(1,-1)'''
        self.IM_vector = {}
        for i in range(1, self.nof_feature+1):
            # np.random.choice([1,-1],size) 隨機二選一 size代表選幾次
            self.IM_vector['feature'+str(i)] = np.random.choice(
                [1, -1], self.nof_dimension).reshape(1, self.nof_dimension).astype(int)

    def init_CIM_vector(self, CIM_vector=None):
        if CIM_vector:
            # test if the CIM vector should be initialized with demanded vectors
            print("Initial CIM with given vectors")
            self.CIM_vector = CIM_vector
            return
        ''' slice continuous signal into self.level parts '''
        # 每往上一個self.level就改 D/2/(self.level-1)個bit

        self.CIM_vector = {}
        nof_change = self.nof_dimension//(2*(self.level-1))
        if self.nof_dimension/2/(self.level-1) != floor(self.nof_dimension/2/(self.level-1)):
            print("warning! D/2/(level-1) is not an integer,", end=' ')
            print(
                "change the dim so that the maximum CIM vector can be orthogonal to the minimum CIM vector")

        self.CIM_vector[0] = np.random.choice(
            [1, -1], self.nof_dimension).reshape(1, self.nof_dimension).astype(int)

        for lev in range(1, self.level):
            # 每個level要改D/2/(level-1)個bit 並且從 D/2/(level-1) * (lev-1)開始改
            # 這裡用到的觀念叫做deep copy 非常重要
            # 只copy value而不是像python assign一樣是share 物件
            self.CIM_vector[lev] = self.CIM_vector[lev-1].copy()
            for index in range(nof_change * (lev-1), nof_change * (lev)):

                self.CIM_vector[lev][0][index] *= -1

    def init_prototype_vector(self):
        '''construct prototype vector'''
        if self.nof_class <= 0:
            print("number of class should pe positive integer!")
            sys.exit(2)
        # Initialize Integer AM and Binary AM
        self.Prototype_vector = {}
        self.Prototype_vector['integer'] = {}
        self.Prototype_vector['binary'] = {}
        # Construct # of class hypervectors for both integer AM and binary AM
        for i in range(0, self.nof_class):
            self.Prototype_vector['integer'][i] = np.zeros(
                [1, self.nof_dimension]).astype(int)
            self.Prototype_vector['binary'][i] = np.zeros(
                [1, self.nof_dimension]).astype(int)

    def encoder_query_vector(self, test_x, query_vector, data=None, retrain=False):
        ''' construct the query vector of each data, and return the predicted result for each data'''
        for feature in range(1, self.nof_feature+1):
            # 因為maximum minimum使用的是train data的資料 要小心超過範圍
            if test_x[feature-1] > self.maximum[feature-1]:
                test_x[feature-1] = self.maximum[feature-1]
            elif test_x[feature-1] < self.minimum[feature-1]:
                test_x[feature-1] = self.minimum[feature-1]

            # 先看這個數值跟這個feature的minimum差多少  算出他的lev 藉此給他相對應的CIM vector
            lev = (test_x[feature-1] - self.minimum[feature-1]
                   )//((self.difference[feature-1])/self.level)

            query_vector += self.IM_vector['feature' +
                                           str(feature)] * self.CIM_vector[0 + lev]

            if feature == 1 and (self.nof_feature % 2) == 0:
                # 因為maximum minimum使用的是train data的資料 要小心超過範圍
                if test_x[feature] > self.maximum[feature]:
                    test_x[feature] = self.maximum[feature]
                elif test_x[feature] < self.minimum[feature]:
                    test_x[feature] = self.minimum[feature]

                # 當self.nof_feature有偶數個 需要補上E1*V1*E2*V2項
                LEV = (test_x[feature] - self.minimum[feature]
                       )//((self.difference[feature]) / (self.level))

                query_vector += self.IM_vector['feature' +
                                               str(feature)] * self.CIM_vector[0+lev] * self.IM_vector['feature' + str(feature+1)] * self.CIM_vector[0+LEV]
        # 這裡有更動 先將query vector做binarize 可能導致accuracy下降
        # query_vector[query_vector > 0] = 1
        # query_vector[query_vector < 0] = -1
        y_pred = self.most_similar_class(query_vector)

        # If this function is called in retrain, we have to return query_vector
        if retrain == True:
            return y_pred, query_vector
        else:
            return y_pred

    def encoder_spatial_vector(self, x, spatial_vector, y):
        '''contruct spatial vector and prototyper vector'''
        for data in range(0, len(x)):
            print("[{}/{}] Dimension:{} Level:{}".format(data,
                                                         len(x), self.nof_dimension, self.level), end='\r')
            # data會是0~最後 字典的key會叫做'featurei'
            for feature in range(1, self.nof_feature + 1):
                # if the value exceeds maximum or falls below minimum, truncate it to maximum/minimum
                if x[data, feature-1] > self.maximum[feature-1]:
                    x[data, feature-1] = self.maximum[feature-1]
                elif x[data, feature-1] < self.minimum[feature-1]:
                    x[data, feature-1] = self.minimum[feature-1]
                # 先看這個數值跟這個feature的minimum差多少  算出他的lev 藉此給他相對應的CIM vector
                lev = (x[data, feature-1] - self.minimum[feature-1]
                       )//((self.difference[feature-1])/(self.level))

                # 每一筆data都會形成一個 Spatial vector S = IM1*CIM1 + IM2*CIM2 + ... 1 2 3代表feature號碼
                spatial_vector[data] += self.IM_vector['feature' +
                                                       str(feature)] * self.CIM_vector[0+lev]

                if feature == 1 and (self.nof_feature % 2) == 0:
                    # 若feature數量為偶數 需要再補上E1*V1*E2*V2這個向量

                    LEV = (x[data, feature] - self.minimum[feature]
                           )//((self.difference[feature]) / (self.level))

                    spatial_vector[data] += self.IM_vector['feature' +
                                                           str(feature)] * self.CIM_vector[0+lev] * self.IM_vector['feature' +
                                                                                                                   str(feature+1)] * self.CIM_vector[0+LEV]

            # 以y的數值 決定這是哪一個class (0~self.nof_class-1)
            whichclass = int(y[data])
            # Binarize spatial_vector (May cause the accuracy to drop)
            # spatial_vector[data][spatial_vector[data] > 0] = 1
            # spatial_vector[data][spatial_vector[data] < 0] = -1

            self.Prototype_vector['integer'][whichclass] += spatial_vector[data]

    def save_model(self, path='HDC_model.pickle'):
        ''' Save the model as pickle file'''
        print('Save model at {}...'.format(path))
        model_dict = {}
        model_dict['IM'] = self.IM_vector
        model_dict['CIM'] = self.CIM_vector
        model_dict['nof_dimension'] = self.nof_dimension
        model_dict['nof_class'] = self.nof_class
        model_dict['nof_feature'] = self.nof_feature
        model_dict['level'] = self.level
        model_dict['PCA'] = self.PCA_projection
        model_dict['BinaryAM'] = self.binaryAM
        model_dict['Prototype_vector'] = self.Prototype_vector
        model_dict['max'] = self.maximum
        model_dict['min'] = self.minimum
        model_dict['difference'] = self.difference
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)

    def load_model(self, path):
        '''load the model'''
        print('Load model from {}...'.format(path))
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        if self.nof_dimension != model_dict['nof_dimension']:
            print("Different nof_dimension between old and loaded model! {}->{}".format(
                self.nof_dimension, model_dict['nof_dimension']))
        if self.nof_class != model_dict['nof_class']:
            print("Different nof_class between old and loaded model! {}->{}".format(
                self.nof_class, model_dict['nof_class']))
        if self.level != model_dict['level']:
            print("Different level between old and loaded model! {}->{}".format(
                self.level, model_dict['level']))
        self.nof_dimension = model_dict['nof_dimension']
        self.nof_class = model_dict['nof_class']
        self.nof_feature = model_dict['nof_feature']
        self.level = model_dict['level']
        self.IM_vector = model_dict['IM']
        self.CIM_vector = model_dict['CIM']
        self.Prototype_vector = model_dict['Prototype_vector']
        self.maximum = model_dict['max']
        self.minimum = model_dict['min']
        self.difference = model_dict['difference']
        self.PCA_projection = model_dict['PCA']
        self.binaryAM = model_dict['BinaryAM']

    def help():
        '''model usage instruction'''
        print("The necessary attribute when you initialize your HDC model like variable = HDC():")
        print("nof_dimension (please enter the dimension of vector)")
        print("nof_class (please enter number of class)")
        print("{:-^40}".format("Usage"))
        print("a=HDC(dim,nof_class,nof_feature,level,PCA_projection)")
        print("a.train(x,y)")
        print("y_pred = a.test(test_x)")
        print("a.accuracy(y_true,y_pred)")
        print("a.retrain_earlystop(test_x,test_y,train_x=None,train_y=None)")
        print("a.result_to_csv('file name')")


if __name__ == "__main__":
    pass
