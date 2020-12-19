#import statements:
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.feature import local_binary_pattern
from tensorflow import keras
import os
from tqdm import tqdm
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from functools import partial
from model import ann
import base64
from flask_socketio import SocketIO, join_room, leave_room, emit

from skimage import feature
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score
from sklearn.ensemble import AdaBoostClassifier
import gc

import pickle

def preprocessingData(file_path):
    #Get the images from a specific path and read them by going over each and every file in the path using tqdm
    if os.path.exists('./dataset.json'):
        with open('./dataset.json', 'r') as f:
            return json.load(f)
    images = []
    for x, y, z in os.walk(file_path):
        for name in tqdm(z):
            images.append(os.path.join(x, name).replace('\\','/'))
    return images

def dataframeCreation(images):
    
    if os.path.exists('./savedata/imagesclass.npy'):
        image_array = np.load('./savedata/images.npy', mmap_mode='c')
        labels = np.load('./savedata/labels.npy', mmap_mode='c')
        return image_array, labels

    sample_points = 16
    radius = 4
    image_array = np.zeros((len(images), 200,200), dtype='float32')
    labels = np.zeros((len(images), 1), dtype=np.dtype('U6'))

    j = 0
    for i in tqdm(images):

        if ".DS_Store" in i:
            continue
        img = cv2.imread(i)
        lbp = LocalBinaryPatterns(sample_points, radius).create_array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 17.
        reshaped = lbp.astype('float32')
        image_array[j] = reshaped
        labels[j] = i.split('/')[2]
        j = j+1

    np.save('./savedata/images.npy', image_array)
    np.save('./savedata/labels.npy', labels)

    return image_array, labels

def dataframeCreationEnsembleModel(images):
    
    if os.path.exists('./savedata/imagesclass.npy'):
        image_array = np.load('./savedata/imagesclass.npy', mmap_mode='c')
        labels = np.load('./savedata/labelsclass.npy', mmap_mode='c')
        return image_array, labels

    sample_points = 16
    radius = 4
    image_array = np.zeros((len(images), 18), dtype='float32')
    labels = np.zeros((len(images), 1), dtype=np.dtype('U6'))

    j = 0
    for i in tqdm(images):

        if ".DS_Store" in i:
            continue
        img = cv2.imread(i)
        lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) 
        reshaped = lbp.astype('float32')
        image_array[j] = reshaped
        labels[j] = i.split('/')[2]
        j = j+1

    np.save('./savedata/imagesclass.npy', image_array)
    np.save('./savedata/labelsclass.npy', labels)

    return image_array, labels


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist = hist / (hist.sum() + eps)
        return hist
    
    def create_array(self, image):
        return local_binary_pattern(image, self.numPoints, self.radius,method="uniform")


def process_image(image, label):
    sample_points, radius = 16,4
    lbp = LocalBinaryPatterns(sample_points, radius).create_array(image.eval())
    return lbp, image

def shuffle_arrays(arrays, shuffle_quant=1):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    max_int = 2**(32 - 1) - 1
    for i in range(shuffle_quant):
        seed = np.random.randint(0, max_int)
        for arr in arrays:
            rstate = np.random.RandomState(seed)
            rstate.shuffle(arr)

print('loading df')
print(os.getcwd())
image_data, image_labels = dataframeCreation(preprocessingData('./dataset/'))
image_data_ensemble, image_labels_ensemble = dataframeCreationEnsembleModel(preprocessingData('./dataset/'))
print(image_data, image_labels)
print('dataframe loaded!')
print(len(image_data_ensemble[0]))


##Data preprocessing:

def ensemble_preprocessing(image_data_ensemble, image_labels_ensemble, testing_size=0.2, optimizeK=True, minK=1,
                           maxK=50, progressObject=None, jobId=-1):
    images_0, labels_0 = image_data_ensemble[(image_labels == '1-10').flatten(), :], image_labels_ensemble[image_labels == '1-10']
    images_1, labels_1 = image_data_ensemble[(image_labels == '11-20').flatten(), :], image_labels_ensemble[image_labels == '11-20']
    images_2, labels_2 = image_data_ensemble[(image_labels == '21-30').flatten(), :], image_labels_ensemble[image_labels == '21-30']
    images_3, labels_3 = image_data_ensemble[(image_labels == '31-40').flatten(), :], image_labels_ensemble[image_labels == '31-40']
    images_4, labels_4 = image_data_ensemble[(image_labels == '41-50').flatten(), :], image_labels_ensemble[image_labels == '41-50']
    images_5, labels_5 = image_data_ensemble[(image_labels == '51-60').flatten(), :], image_labels_ensemble[image_labels == '51-60']
    images_6, labels_6 = image_data_ensemble[(image_labels == '61-70').flatten(), :], image_labels_ensemble[image_labels == '61-70']
    images_7, labels_7 = image_data_ensemble[(image_labels == '71-80').flatten(), :], image_labels_ensemble[image_labels == '71-80']
    images_8, labels_8 = image_data_ensemble[(image_labels == '81-116').flatten(), :], image_labels_ensemble[image_labels == '81-116']


    for i in range (labels_0.size):
        labels_0[i] = 'young'


    for i in range(labels_1.size):
        labels_1[i] = 'young'

    for i in range(labels_2.size):
        labels_2[i] = 'young'

    for i in range(labels_3.size):
        labels_3[i] = 'young'

    for i in range(labels_4.size):
        labels_4[i] = 'old'

    for i in range(labels_5.size):
        labels_5[i] = 'old'

    for i in range(labels_6.size):
        labels_6[i] = 'old'

    for i in range(labels_7.size):
        labels_7[i] = 'old'

    for i in range(labels_8.size):
        labels_8[i] = 'old' 

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(images_0, labels_0, test_size=testing_size)
    X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(images_1, labels_1, test_size=testing_size)
    X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(images_2, labels_2, test_size=testing_size)
    X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(images_3, labels_3, test_size=testing_size)
    X_train_40, X_test_40, y_train_40, y_test_40 = train_test_split(images_4, labels_4, test_size=testing_size)
    X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(images_5, labels_5, test_size=testing_size)
    X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(images_6, labels_6, test_size=testing_size)
    X_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(images_7, labels_7, test_size=testing_size)
    X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(images_8, labels_8, test_size=testing_size)

    X_train = np.concatenate((X_train_0, X_train_10, X_train_20, X_train_30, X_train_40, X_train_50, X_train_60,  X_train_70, X_train_80), axis=0)
    y_train = np.concatenate((y_train_0, y_train_10, y_train_20, y_train_30, y_train_40, y_train_50, y_train_60, y_train_70, y_train_80), axis=0)


    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train = lb.transform(y_train)
    shuffle_arrays([X_train, y_train])

    X_test = np.concatenate((X_test_0, X_test_10, X_test_20, X_test_30, X_test_40, X_test_50, X_test_60,  X_test_70, X_test_80), axis=0)
    y_test = np.concatenate((y_test_0, y_test_10, y_test_20, y_test_30, y_test_40, y_test_50, y_test_60,  y_test_70, y_test_80), axis=0)
    y_test = lb.transform(y_test)
    shuffle_arrays([X_test,y_test])




    scores = {}
    scores_list = []

    if optimizeK:
        k_range = range(minK, maxK + 1)
        for k in tqdm(k_range):
            knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance',algorithm='ball_tree', p=3)
            out = np.concatenate(y_train).ravel()
            knn.fit(X_train, out)
            y_pred = knn.predict(X_test)
            scores[k] = metrics.accuracy_score(y_test, y_pred)
            scores_list.append(metrics.accuracy_score(y_test, y_pred))
            if progressObject != None:
                progressObject['jobProgress'][jobId] = min((k_range.index(k) + 1.0) / len(k_range), 0.99)

        k_optimal = scores_list.index(max(scores_list))
    else:
        k_optimal = minK

    model = KNeighborsClassifier(n_neighbors= k_optimal, weights = 'distance', algorithm='ball_tree', p=3)
    model2 = svm.SVC(kernel='rbf')
    model3=VotingClassifier(estimators=[('KNN',model), ('SVM',model2)], voting='hard')

    model3.fit(X_train, y_train.ravel())

    #Train the model on the entire training set

    print(X_train.shape)
    print(y_train.shape)

    # model3.fit(X_train, y_train)
    y_pred = model3.predict(X_test)

    # kfold_ada = model_selection.KFold(n_splits=10, random_state=10)
    # model_ada = AdaBoostClassifier(n_estimators=30, random_state=10)
    # model_ada.fit(X_train, y_train)
    # results_ada = model_selection.cross_val_score(model_ada, X_train, y_train, cv=kfold_ada)
    # rs = model_ada.predict(X_test)
    results_adas = classification_report(y_test, y_pred, output_dict=True)
    # print(results_ada.mean())
    print(results_adas)

    return model3, results_adas, k_optimal
    # todo: return ensemble model, and all scores


def customEnsembleProcessing(images_young, images_adult, images_middle, images_old, testing_size=0.2, optimizeK=True, minK=1,
                           maxK=50, progressObject=None, jobId=-1):

    young_labels = np.zeros((len(images_young)+len(images_old),1))
    old_labels = np.ones((len(images_adult)+len(images_middle),1))

    young_train = np.concatenate((images_young, images_adult), axis=0)
    old_train = np.concatenate((images_middle, images_old), axis=0)

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(young_train, young_labels, test_size=testing_size)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(old_train, old_labels, test_size=testing_size)

    X_train = np.concatenate(
        (X_train_0, X_train_1),
        axis=0)
    y_train = np.concatenate(
        (y_train_0, y_train_1),
        axis=0)

    # lb = LabelBinarizer()
    # lb.fit(y_train)
    # y_train = lb.transform(y_train)
    shuffle_arrays([X_train, y_train])

    X_test = np.concatenate(
        (X_test_0, X_test_1), axis=0)
    y_test = np.concatenate(
        (y_test_0, y_test_1), axis=0)
    # y_test = lb.transform(y_test)
    shuffle_arrays([X_test, y_test])

    scores = {}
    scores_list = []

    if optimizeK:
        k_range = range(minK, maxK + 1)
        for k in tqdm(k_range):
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='ball_tree', p=3)
            out = np.concatenate(y_train).ravel()
            knn.fit(X_train, out)
            y_pred = knn.predict(X_test)
            scores[k] = metrics.accuracy_score(y_test, y_pred)
            scores_list.append(metrics.accuracy_score(y_test, y_pred))
            if progressObject != None:
                progressObject['jobProgress'][jobId] = min((k_range.index(k) + 1.0) / len(k_range), 0.99)

        k_optimal = scores_list.index(max(scores_list))
    else:
        k_optimal = minK

    model = KNeighborsClassifier(n_neighbors=k_optimal, weights='distance', algorithm='ball_tree', p=3)
    model2 = svm.SVC(kernel='rbf')
    # model3=VotingClassifier(estimators=[('KNN',model), ('SVM',model2)], voting='hard')

    model.fit(X_train, y_train.ravel())

    # Train the model on the entire training set

    print(X_train.shape)
    print(y_train.shape)

    # model3.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_adas = classification_report(y_test, y_pred, output_dict=True)
    # print(results_ada.mean())
    print(results_adas)

    return model, results_adas, k_optimal
    # todo: return ensemble model, and all scores


def label_netowork_young(labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_7, labels_8):

    for i in range (labels_0.size):
        labels_0[i] = 'young'

    for i in range(labels_1.size):
        labels_1[i] = 'young'

    for i in range(labels_2.size):
        labels_2[i] = 'adult'

    for i in range(labels_3.size):
        labels_3[i] = 'adult'

    for i in range(labels_4.size):
        labels_4[i] = 'Not Young'

    for i in range(labels_5.size):
        labels_5[i] = 'Not Young'

    for i in range(labels_6.size):
        labels_6[i] = 'Not Young'

    for i in range(labels_7.size):
        labels_7[i] = 'Not Young'

    for i in range(labels_8.size):
        labels_8[i] = 'Not Young' 

def label_netowork_old(labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_7, labels_8):
    
    for i in range (labels_0.size):
        labels_0[i] = 'Not Old'

    for i in range(labels_1.size):
        labels_1[i] = 'Not Old'

    for i in range(labels_2.size):
        labels_2[i] = 'Not Old'

    for i in range(labels_3.size):
        labels_3[i] = 'Not Old'

    for i in range(labels_4.size):
        labels_4[i] = 'Middle Aged'

    for i in range(labels_5.size):
        labels_5[i] = 'Middle Aged'

    for i in range(labels_6.size):
        labels_6[i] = 'Old'

    for i in range(labels_7.size):
        labels_7[i] = 'Old'

    for i in range(labels_8.size):
        labels_8[i] = 'Old' 



def preprocess_labels_ann_neural(label, testing_ratio=0.2):
    images_0, labels_0 = image_data[(image_labels == '1-10').flatten(), :], image_labels[image_labels == '1-10']
    images_1, labels_1 = image_data[(image_labels == '11-20').flatten(), :], image_labels[image_labels == '11-20']
    images_2, labels_2 = image_data[(image_labels == '21-30').flatten(), :], image_labels[image_labels == '21-30']
    images_3, labels_3 = image_data[(image_labels == '31-40').flatten(), :], image_labels[image_labels == '31-40']
    images_4, labels_4 = image_data[(image_labels == '41-50').flatten(), :], image_labels[image_labels == '41-50']
    images_5, labels_5 = image_data[(image_labels == '51-60').flatten(), :], image_labels[image_labels == '51-60']
    images_6, labels_6 = image_data[(image_labels == '61-70').flatten(), :], image_labels[image_labels == '61-70']
    images_7, labels_7 = image_data[(image_labels == '71-80').flatten(), :], image_labels[image_labels == '71-80']
    images_8, labels_8 = image_data[(image_labels == '81-116').flatten(), :], image_labels[image_labels == '81-116']

    if label == 'Young':
        label_netowork_young(labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_7, labels_8)
    
    if label == 'Old':
        label_netowork_old(labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, labels_7, labels_8)

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(images_0, labels_0, test_size=testing_ratio)
    X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(images_1, labels_1, test_size=testing_ratio)
    X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(images_2, labels_2, test_size=testing_ratio)
    X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(images_3, labels_3, test_size=testing_ratio)
    X_train_40, X_test_40, y_train_40, y_test_40 = train_test_split(images_4, labels_4, test_size=testing_ratio)
    X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(images_5, labels_5, test_size=testing_ratio)
    X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(images_6, labels_6, test_size=testing_ratio)
    X_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(images_7, labels_7, test_size=testing_ratio)
    X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(images_8, labels_8, test_size=testing_ratio)

    X_train = np.concatenate((X_train_0, X_train_10, X_train_20, X_train_30, X_train_40, X_train_50, X_train_60,  X_train_70, X_train_80), axis=0)
    y_train = np.concatenate((y_train_0, y_train_10, y_train_20, y_train_30, y_train_40, y_train_50, y_train_60, y_train_70, y_train_80), axis=0)
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train = lb.transform(y_train)
    shuffle_arrays([X_train, y_train])
    X_test = np.concatenate((X_test_0, X_test_10, X_test_20, X_test_30, X_test_40, X_test_50, X_test_60,  X_test_70, X_test_80), axis=0)
    y_test = np.concatenate((y_test_0, y_test_10, y_test_20, y_test_30, y_test_40, y_test_50, y_test_60,  y_test_70, y_test_80), axis=0)
    y_test = lb.transform(y_test)
    shuffle_arrays([X_test,y_test])
    return X_train, y_train, X_test, y_test

def train_network(X_train, y_train, X_test, y_test, weights, nb_of_outputs,save_name, nb_of_epochs=10000):
    nn = ann(weights= weights, nb_of_outputs =nb_of_outputs, X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, entropy= 'categorical_crossentropy', nb_of_epochs=nb_of_epochs )
    exact_acc, one_off, two_off, time_train, hist = nn.modelTrain(weights=weights, nb_of_outputs=nb_of_outputs , X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, entropy = 'categorical_crossentropy', nb_of_epochs = 10000, save_name=save_name)
    return exact_acc, one_off, two_off, time_train, hist

    
def predict_final_label_ensemble(X_test_ens, X_test, ensemble_pretrained=True, young_pretrained=True, old_pretrained=True):
    #load ensemble model:

    if ensemble_pretrained:
        ensemble_name = './savedata/pretrained_ensemble'
    else:
        ensemble_name = './savedata/user_ensemble'

    if young_pretrained:
        young_name = './savedata/pretrained_young.h5'
    else:
        young_name = './savedata/user_young.h5'

    if old_pretrained:
        old_name = './savedata/pretrained_old.h5'
    else:
        old_name = './savedata/user_old.h5'


    model = pickle.load(open(ensemble_name, 'rb'))
    first_X = X_test_ens.reshape(1,-1)
    ann_X = X_test.reshape(1,200,200)
    y_pred=model.predict(first_X)

    #if y_pred of ensemble model is young : 0
    if y_pred==1:
        print("Ensemble model classified image as Young")
    
        #Retreive loaded young neural network: 
        #Assuming that the it is saved in mybaby.h5: need to pretrain and load each of the young and old neural networks and keep them saved
        model_young = load_model(young_name)
        final_predicts = model_young.predict(ann_X)
        if np.argmax(final_predicts) == 0:
            #Classified as Not Young:
            #load old model
            model_old = load_model(old_name)
            final_predicts = model_old.predict(ann_X)
            if np.argmax(final_predicts) == 1:
                #predicted as not old:
                return "Error state"

            else:
                if np.argmax(final_predicts) == 0:
                    return 'MiddleAge'
                else:
                    return 'Old'
        else:
            if np.argmax(final_predicts) == 1:
                return 'Adult'
            else:
                return 'Young'
    
    if y_pred==0:
        #old
        print("Ensemble model classified image as Old")
    
        #Retreive loaded old neural network: 
        #Assuming that the it is saved in mybaby.h5: need to pretrain and load each of the young and old neural networks and keep them saved
        model_old = load_model(old_name)
        final_predicts = model_old.predict(ann_X)
        if np.argmax(final_predicts) == 1:
            #Classified as Not old:
            #load young model
            model_young = load_model(young_name)
            final_predicts = model_young.predict(ann_X)
            if np.argmax(final_predicts) == 0:
                #predicted as not young:
                return "Error state"
            else:
                if np.argmax(final_predicts) == 1:
                    return 'Adult'
                else:
                    return 'Young'
        else:
            if np.argmax(final_predicts) == 0:
                return 'MiddleAge'
            else:
                return 'Old'


def predict_final_label_classes(X_test):
    model = load_model('./savedata/classes.h5')
    ann_X = X_test.reshape(1,200,200)
    result = model.predict(ann_X)

    index = np.argmax(result)
    if index == 0:
        return '1-10'
    elif index == 1:
        return '11-20'
    elif index == 2:
        return '21-30'
    elif index == 3:
        return '31-40'
    elif index == 4:
        return '41-50'
    elif index == 5:
        return '51-60'
    elif index == 6:
        return '61-70'
    elif index == 7:
        return '71-80'
    else:
        return '81-'

def initializeML(initial_dataset_path='./dataset'):
    sharedObject = {}
    model_type = 'ensemble'
    model_ensemble = 'pretrained_ensemble'
    model_old = 'pretrained_old'
    model_young = 'pretrained_young'
    model_classes = 'pretrained_classes'

    if os.path.exists('./savedata/config.json'):
        print('Found previously saved config!')
        with open('./savedata/config.json') as f:
            sharedObject = json.load(f)
    else:
        # load pre-packages model, and calculate its initial scores based on the dataset
        print('No config found, assuming first launch!')

        # images = preprocessingData(initial_dataset_path)
        # X_young, y_young, X_old, y_old, _ = dataframeCreation(images)
        # eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train(X_young, y_young, X_old, y_old,
        #                                                                           k_cross_validation_ratio=5,
        #                                                                           testing_size=0.2, optimal_k=True,
        #                                                                           min_range_k=1, max_range_k=100)
        # test_score, conf_rep = test(X_train, y_train, X_test, y_test, modelName=model_name)

        # setup the initial config
        sharedObject['model_type'] = model_type
        sharedObject['model_ensemble'] = model_ensemble
        sharedObject['model_young'] = model_young
        sharedObject['model_old'] = model_old
        sharedObject['model_classes'] = model_classes

        with open('./savedata/ensemble.json', 'r') as f:
            ensembleScores = json.load(f)

        with open('./savedata/youngnn.json', 'r') as f:
            youngScores = json.load(f)

        with open('./savedata/oldnn.json', 'r') as f:
            oldScores = json.load(f)

        sharedObject['ensemble_pretrained'] = ensembleScores
        sharedObject['youngNNPretrained'] = youngScores
        sharedObject['oldNNPretrained'] = oldScores

        with open('./savedata/config.json', 'w') as f:
            json.dump(sharedObject, f)

        # sharedObject['p_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'],
        #                                   'acc': conf_rep['accuracy'], 'test_score': test_score}
        # sharedObject['p_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0], 'test_nbr': X_test.shape[0]}



    return sharedObject


def locateFace(img):
    # Load the opencv cascade for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces

    # Get coordinates of every face
    for (x, y, w, h) in faces:
        cv2.rectangle(np.copy(img), (x, y), (x + w, y + h), (255, 0, 0), 2)
    if faces == ():
        x = -1
        y = -1
        w = -1
        h = -1
    return faces, x, y, w, h

def createInputFromBase64(base64Image):
    encoded_data = base64Image.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    # the parameters of the LBP algo
    # higher = more time required
    sample_points = 16
    radius = 4
    images_crop = []
    faces, x, y, w, h = locateFace(img)
    if not faces == ():
        crop_img = img[y:y + h, x:x + w]
        resized = cv2.resize(crop_img, (200,200), interpolation=cv2.INTER_AREA)
        lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
        X_ens = lbp.astype('float32')

        lbp = LocalBinaryPatterns(sample_points, radius).create_array(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)) / 17.
        X = lbp.astype('float32')

        return X_ens,X

    else:
        print("Error! No face was detected. Please try again with a clearer picture")

    return None, None

def predictFromBase64(base64Image):
    #get the current model and extract the data values from the images
    #predict the labels using the predict method
    # model = getCurrentModel()
    with open('./savedata/config.json') as f:
        sharedObject = json.load(f)
    X_ens, X = createInputFromBase64(base64Image)

    if X_ens is None:
        return 'No Face!'

    if sharedObject['model_type'] == 'ensemble':
        ensemble_pretrained = sharedObject['model_ensemble'] == 'pretrained_ensemble'
        young_pretrained = sharedObject['model_young'] = 'pretrained_young'
        old_pretrained = sharedObject['model_old'] = 'pretrained_old'
        return predict_final_label_ensemble(X_ens, X, ensemble_pretrained, young_pretrained, old_pretrained)
    else:
        return predict_final_label_classes(X)


def createEnsembleDFFromBase64(images, progressObject, jobId):

    sample_points = 16
    radius = 4
    image_array = np.zeros((len(images), 18), dtype='float32')
    progressObject['jobProgress'][jobId] = 0.0

    for base64Image in images:
        encoded_data = base64Image.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces, x,y,w,h = locateFace(img)
        if not faces == ():
            crop_img = img[y:y+h, x:x+w]
            resized = cv2.resize(crop_img, (200, 200), interpolation=cv2.INTER_AREA)
            lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
            reshaped = lbp.astype('float32')
            image_array[j] = reshaped
            j = j + 1

        if progressObject != None:
            progressObject['jobProgress'][jobId] = min((images.index(base64Image)+1.0)/len(images), 0.99)

    return image_array

def creatNNDFFromBase64(images, progressObject, jobId):

    sample_points = 16
    radius = 4
    image_array = np.zeros((len(images), 200,200), dtype='float32')
    progressObject['jobProgress'][jobId] = 0.0

    for base64Image in images:
        encoded_data = base64Image.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces, x,y,w,h = locateFace(img)
        if not faces == ():
            crop_img = img[y:y+h, x:x+w]
            resized = cv2.resize(crop_img, (200, 200), interpolation=cv2.INTER_AREA)
            lbp = LocalBinaryPatterns(sample_points, radius).create_array(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))/17.
            reshaped = lbp.astype('float32')
            image_array[j] = reshaped
            j = j + 1

        if progressObject != None:
            progressObject['jobProgress'][jobId] = min((images.index(base64Image)+1.0)/len(images), 0.99)

    return image_array


def createDFANNTrainEnsemble(type, young, adult, middle, old, testing_ratio=0.2):
    if type == 'Young':
        y_adult=np.zeros((len(adult), 3))
        y_adult[:,1]=1

        notyoung =np.concatenate((middle, old),axis=0)
        y_notyoung=np.zeros((len(middle)+len(old),3))
        y_notyoung[:,0]=1

        y_young=np.zeros((len(young),3))
        y_young[:,2]=1

        X_train_adult, X_test_adult, y_train_adult, y_test_adult = train_test_split(adult, y_adult, test_size=testing_ratio)
        X_train_notyoung, X_test_notyoung, y_train_notyoung, y_test_notyoung = train_test_split(notyoung, y_notyoung, test_size=testing_ratio)
        X_train_young, X_test_young, y_train_young, y_test_young = train_test_split(young, y_young, test_size=testing_ratio)

        X_train = np.concatenate((X_train_adult, X_train_notyoung, X_train_young), axis=0)
        y_train = np.concatenate((y_train_adult, y_train_notyoung, y_train_young), axis=0)

        shuffle_arrays([X_train, y_train])

        X_test = np.concatenate((X_test_adult, X_test_notyoung, X_test_young), axis=0)
        y_test = np.concatenate((y_test_adult, y_test_notyoung, y_test_young), axis=0)

        shuffle_arrays([X_test, y_test])

        return X_train, y_train, X_test, y_test
    else:
        y_middle = np.zeros((len(middle), 3))
        y_middle[:, 0] = 1

        notold = np.concatenate((young, adult), axis=0)
        y_notold = np.zeros((len(young) + len(adult), 3))
        y_notold[:, 1] = 1

        y_old = np.zeros((len(old), 3))
        y_old[:, 2] = 1

        X_train_middle, X_test_middle, y_train_middle, y_test_middle = train_test_split(middle, y_middle,
                                                                                    test_size=testing_ratio)
        X_train_notold, X_test_notold, y_train_notold, y_test_notold = train_test_split(notold, y_notold,
                                                                                                test_size=testing_ratio)
        X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(young, y_old,
                                                                                    test_size=testing_ratio)

        X_train = np.concatenate((X_train_middle, X_train_notold, X_train_old), axis=0)
        y_train = np.concatenate((y_train_middle, y_train_notold, y_train_old), axis=0)

        shuffle_arrays([X_train, y_train])

        X_test = np.concatenate((X_test_middle, X_test_notold, X_test_old), axis=0)
        y_test = np.concatenate((y_test_middle, y_test_notold, y_test_old), axis=0)

        shuffle_arrays([X_test, y_test])

        return X_train, y_train, X_test, y_test


def performJob(job, sharedObject):
    # received a job from backend, view the task and execute it
    jobType = job['type']
    print(type(sharedObject))
    if jobType == 'PREDICT':
        # predict image label
        image = job['image']
        labels = predictFromBase64(image)
        print('final results:', labels)
        # socket.emit('predict-result', labels, room=job['client'])
        # print('prediction finished', labels)
        sharedObject['jobProgress'][job['jobID']] = 1.0
        sharedObject['jobResults'][job['jobID']] = {'label': labels}
    if jobType == 'TRAIN':
        print(job)
        if job['trainType'] == 'reset':
            model_ensemble = 'pretrained_ensemble'
            model_old = 'pretrained_old'
            model_young = 'pretrained_young'
            model_classes = 'pretrained_classes'

            newConfig = {}
            newConfig['model_ensemble'] = model_ensemble
            newConfig['model_young'] = model_young
            newConfig['model_old'] = model_old
            newConfig['model_classes'] = model_classes
            if job['resetType'] == 'ensemble':
                model_type = 'ensemble'
            else:
                model_type = 'classes'
            newConfig['model_type'] = model_type

            with open('./savedata/ensemble.json', 'r') as f:
                ensembleScores = json.load(f)

            with open('./savedata/youngnn.json', 'r') as f:
                youngScores = json.load(f)

            with open('./savedata/oldnn.json', 'r') as f:
                oldScores = json.load(f)

            newConfig['ensemble_pretrained'] = ensembleScores
            newConfig['youngNNPretrained'] = youngScores
            newConfig['oldNNPretrained'] = oldScores

            with open('./savedata/config.json', 'w') as f:
                json.dump(newConfig, f)

            sharedObject['jobProgress'][job['jobID']] = 1.0

        elif job['trainType'] == 'ensemble':
            if not job['trainInitial']:
                model_ensemble = 'pretrained_ensemble'
                sharedObject['model_ensemble'] = model_ensemble

                with open('./savedata/config.json', 'r+') as f:
                        config = json.load(f)
                        f.seek(0)
                        config['model_ensemble'] = model_ensemble
                        json.dump(config, f)
                        f.truncate()
            else:
                model_ensemble = 'user_ensemble'
                ## TRAIN SVM KNN

                if job['customData']:
                    young_images = createEnsembleDFFromBase64(job['youngImages'], sharedObject, job['jobID'])
                    adult_images = createEnsembleDFFromBase64(job['adultImages'], sharedObject, job['jobID'])
                    middle_aged = createEnsembleDFFromBase64(job['middleAgedImages'], sharedObject, job['jobID'])
                    old_images = createEnsembleDFFromBase64(job['oldImages'], sharedObject, job['jobID'])
                    model, results, k = customEnsembleProcessing(young_images, adult_images, middle_aged, old_images,
                                                                 testing_size=job['testingRatio'],
                                                                 optimizeK=job['optimizeK'],
                                                                 minK=job['minK'], maxK=job['maxK'], progressObject=sharedObject,
                                                                jobId=job['jobID'])
                    pickle.dump(model, open('./savedata/user_ensemble', 'wb'))

                    data = {
                        'KNNScore': results, 'K': k
                    }
                    sharedObject['ensemble'] = data
                    sharedObject['model_ensemble'] = model_ensemble

                    with open('./savedata/config.json', 'r+') as f:
                        config = json.load(f)
                        f.seek(0)
                        config['model_ensemble'] = model_ensemble
                        config['ensemble']=data
                        json.dump(config, f)
                        f.truncate()

                else:
                    model, results, k = ensemble_preprocessing(image_data_ensemble, image_labels_ensemble,
                                                               testing_size=job['testingRatio'], optimizeK=job['optimizeK'],
                                                               minK=job['minK'], maxK=job['maxK'], progressObject=sharedObject,
                                                               jobId=job['jobID'])
                    pickle.dump(model, open('./savedata/user_ensemble', 'wb'))

                    data = {
                        'KNNScore': results, 'K': k
                    }
                    sharedObject['ensemble'] = data
                    sharedObject['model_ensemble'] = model_ensemble

                    with open('./savedata/config.json', 'r+') as f:
                        config = json.load(f)
                        f.seek(0)
                        config['model_ensemble'] = model_ensemble
                        config['ensemble']=data
                        json.dump(config, f)
                        f.truncate()

            if not job['trainYoung']:
                model_young = 'pretrained_young'
                sharedObject['model_young'] = model_young

                with open('./savedata/config.json', 'r+') as f:
                    config = json.load(f)
                    f.seek(0)
                    config['model_young'] = model_young
                    json.dump(config, f)
                    f.truncate()
            else:
                model_young = 'user_young'
                ## TRAIN YOUNG

                if job['customData']:
                    youngImages = creatNNDFFromBase64(job['youngImages'], sharedObject, job['jobID'])
                    adultImages = creatNNDFFromBase64(job['adultImages'], sharedObject, job['jobID'])
                    middleAgedImages = creatNNDFFromBase64(job['middleAgedImages'], sharedObject, job['jobID'])
                    oldImages = creatNNDFFromBase64(job['oldImages'], sharedObject, job['jobID'])

                    X_train, y_train, X_test, y_test = createDFANNTrainEnsemble('Young', youngImages, adultImages, middleAgedImages, oldImages, testing_ratio=job['testingRatio'])
                    exact_acc, one_off, two_off, time_train, hist = train_network(X_train, y_train, X_test, y_test,
                                                                                  {0: 1.7, 1: 1., 2: 2}, 3,
                                                                                  './savedata/user_young.h5')

                    data = {
                        'accuracy': exact_acc,
                        'oneOff': one_off,
                        'twoOff': two_off,
                        'trainTime': time_train,
                        'history': hist.history
                    }

                    sharedObject['youngNN'] = data
                    sharedObject['model_young'] = model_young

                    with open('./savedata/config.json', 'r+') as f:
                        config = json.load(f)
                        f.seek(0)
                        config['model_young'] = model_young
                        config['youngNN'] = data
                        json.dump(config, f)
                        f.truncate()
                else:
                    X_train, y_train, X_test, y_test = preprocess_labels_ann_neural('Young', testing_ratio=job['testingRatio'])
                    exact_acc, one_off, two_off, time_train, hist = train_network(X_train, y_train, X_test, y_test, {0: 1.7, 1:1., 2:2}, 3, './savedata/user_young.h5')

                    data = {
                        'accuracy': exact_acc,
                        'oneOff': one_off,
                        'twoOff': two_off,
                        'trainTime': time_train,
                        'history': hist.history
                    }

                    sharedObject['youngNN'] = data
                    sharedObject['model_young'] =model_young

                    with open('./savedata/config.json', 'r+') as f:
                        config = json.load(f)
                        f.seek(0)
                        config['model_young'] = model_young
                        config['youngNN']=data
                        json.dump(config, f)
                        f.truncate()

            if not job['trainOld']:
                model_old = 'pretrained_old'
                sharedObject['model_old'] = model_old

                with open('./savedata/config.json', 'r+') as f:
                    config = json.load(f)
                    f.seek(0)
                    config['model_old'] = model_old
                    json.dump(config, f)
                    f.truncate()

            else:
                model_old = 'user_old'
                ## TRAIN OLD KNN

                if job['customData']:
                    youngImages = creatNNDFFromBase64(job['youngImages'], sharedObject, job['jobID'])
                    adultImages = creatNNDFFromBase64(job['adultImages'], sharedObject, job['jobID'])
                    middleAgedImages = creatNNDFFromBase64(job['middleAgedImages'], sharedObject, job['jobID'])
                    oldImages = creatNNDFFromBase64(job['oldImages'], sharedObject, job['jobID'])

                    X_train, y_train, X_test, y_test = createDFANNTrainEnsemble('Old', youngImages, adultImages,
                                                                                middleAgedImages, oldImages,
                                                                                testing_ratio=job['testingRatio'])

                    exact_acc, one_off, two_off, time_train, hist = train_network(X_train, y_train, X_test, y_test,
                                                                                  {0: 1.5, 1: 1., 2: 6.}, 3,
                                                                                  './savedata/user_old.h5')

                    data = {
                        'accuracy': exact_acc,
                        'oneOff': one_off,
                        'twoOff': two_off,
                        'trainTime': time_train,
                        'history': hist.history
                    }

                    sharedObject['oldNN'] = data
                    sharedObject['model_old'] = model_old

                    with open('./savedata/config.json', 'r+') as f:
                        config = json.load(f)
                        f.seek(0)
                        config['model_old'] = model_old
                        config['oldNN'] = data
                        json.dump(config, f)
                        f.truncate()

                else:
                    X_train, y_train, X_test, y_test = preprocess_labels_ann_neural('Old',
                                                                                    testing_ratio=job['testingRatio'])
                    exact_acc, one_off, two_off, time_train, hist = train_network(X_train, y_train, X_test, y_test,
                                                                                  {0: 1.5, 1:1., 2:6.}, 3,
                                                                                  './savedata/user_old.h5')

                    data = {
                        'accuracy': exact_acc,
                        'oneOff': one_off,
                        'twoOff': two_off,
                        'trainTime': time_train,
                        'history': hist.history
                    }

                    sharedObject['oldNN'] = data
                    sharedObject['model_old'] = model_old

                    with open('./savedata/config.json', 'r+') as f:
                        config = json.load(f)
                        f.seek(0)
                        config['model_old'] = model_old
                        config['oldNN'] = data
                        json.dump(config, f)
                        f.truncate()

            sharedObject['jobProgress'][job['jobID']] = 1.0

    #
    #         eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train2(X, y, k_cross_validation_ratio=5,
    #                                                                                    testing_size=job['test_ratio'],
    #                                                                                    optimal_k=job['optimizeK'],
    #                                                                                    min_range_k=job['minK'],
    #                                                                                    max_range_k=maxK,
    #                                                                                    model_name=model_name,
    #                                                                                    progressObject=sharedObject,
    #                                                                                    jobId=job['jobID'])
    #         test_score, conf_rep = test(X_train, y_train, X_test, y_test, modelName=model_name)
    #
    #         sharedObject['model_name'] = model_name
    #         sharedObject['u_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'],
    #                                           'acc': conf_rep['accuracy'], 'test_score': test_score}
    #         sharedObject['u_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0],
    #                                           'test_nbr': X_test.shape[0]}
    #
    #         with open('./config.json', 'r+') as f:
    #             config = json.load(f)
    #             f.seek(0)
    #             config['model_name'] = model_name
    #             config['u_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'],
    #                                         'acc': conf_rep['accuracy'], 'test_score': test_score}
    #             config['u_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0], 'test_nbr': X_test.shape[0]}
    #             json.dump(config, f)
    #             f.truncate()
    #         sharedObject['jobProgress'][job['jobID']] = 1.0
    #         sharedObject['isTraining'] = False
    #         print('finishedTraining!')
    #
    #
    #     elif job['trainType'] == 'custom':
    #         # train a custom model
    #         sharedObject['isTraining'] = True
    #         sharedObject['trainingId'] = job['jobID']
    #         print('Creating model from custom dataset!')
    #         model_name = "user_knn_model"
    #
    #         imagesYoung = job['youngPics']
    #         X_young, y_young = createTrainDFFromBase64(imagesYoung, True, sharedObject, job['jobID'])
    #
    #         imagesOld = job['oldPics']
    #         X_old, y_old = createTrainDFFromBase64(imagesOld, False, sharedObject, job['jobID'])
    #
    #         optimize = job['optimizeK']
    #         if optimize:
    #             maxK = job['maxK']
    #         else:
    #             maxK = 100
    #         eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train(X_young, y_young, X_old, y_old,
    #                                                                                   k_cross_validation_ratio=5,
    #                                                                                   testing_size=job['test_ratio'],
    #                                                                                   optimal_k=job['optimizeK'],
    #                                                                                   min_range_k=job['minK'],
    #                                                                                   max_range_k=maxK,
    #                                                                                   model_name=model_name,
    #                                                                                   progressObject=sharedObject,
    #                                                                                   jobId=job['jobID'])
    #         test_score, conf_rep = test(X_train, y_train, X_test, y_test, modelName=model_name)
    #
    #         print(conf_rep)
    #         sharedObject['model_name'] = model_name
    #         sharedObject['u_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'],
    #                                           'acc': conf_rep['accuracy'], 'test_score': test_score}
    #         sharedObject['u_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0],
    #                                           'test_nbr': X_test.shape[0]}
    #
    #         with open('./config.json', 'r+') as f:
    #             config = json.load(f)
    #             f.seek(0)
    #             config['model_name'] = model_name
    #             config['u_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'],
    #                                         'acc': conf_rep['accuracy'], 'test_score': test_score}
    #             config['u_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0], 'test_nbr': X_test.shape[0]}
    #             json.dump(config, f)
    #             f.truncate()
    #         sharedObject['jobProgress'][job['jobID']] = 1.0
    #         sharedObject['isTraining'] = False
    #         print('finishedTraining!')

# # if not os.path.exists('./savedata/pretrained_ensemble'):
# model, results, k = ensemble_preprocessing(image_data_ensemble, image_labels_ensemble)
# pickle.dump(model, open('./savedata/pretrained_ensemble', 'wb'))
# output = {
#     'KNNScore': results,
#     'K': k
# }
#
# with open('./savedata/ensemble.json', 'w') as f:
#     json.dump(output, f)
#
# sharedObject = {}
# X_train, y_train, X_test, y_test = preprocess_labels_ann_neural('Young')
# exact_acc, one_off, two_off, time_train, hist = train_network(X_train, y_train, X_test, y_test, {0: 1.7, 1:1., 2:2}, 3, './savedata/pretrained_young.h5')
#
# sharedObject['youngExact'] = exact_acc
# sharedObject['youngOneOff'] = one_off
# sharedObject['youngTwoOff'] = two_off
# sharedObject['youngTrainTime'] = time_train
# sharedObject['youngHist'] = hist.history
#
# X_train, y_train, X_test, y_test = preprocess_labels_ann_neural('Old')
# exact_acc, one_off, two_off, time_train, hist = train_network(X_train, y_train, X_test, y_test, {0: 1.5, 1:1., 2:6.}, 3, './savedata/pretrained_old.h5')
#
# sharedObject['oldExact'] = exact_acc
# sharedObject['oldOneOff'] = one_off
# sharedObject['oldTwoOff'] = two_off
# sharedObject['oldTrainTime'] = time_train
# sharedObject['oldHist'] = hist.history
#
# with open('./savedata/oldnn.json', 'w') as f:
#     json.dump(sharedObject, f)

#
# images = []
# for x, y, z in os.walk('./dataset'):
#     for name in tqdm(z):
#         images.append(os.path.join(x, name).replace('\\','/'))
#
#
# correct = 0
# incorrect = 0
# errors = 0
#
# for i in tqdm(images):
#     gc.collect()
#     img = cv2.imread(i)
#     lbp = LocalBinaryPatterns(16, 4).create_array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 17.
#     reshaped_ann = lbp.astype('float32').reshape(1,200,200)
#
#     lbp = LocalBinaryPatterns(16, 4).describe(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#     reshaped_ens = lbp.astype('float32').reshape(1,-1)
#
#     model = pickle.load(open('./savedata/pretrained_ensemble', 'rb'))
#
#     label = predict_final_label(model, reshaped_ens, reshaped_ann)
#
#     if label == 'Error state':
#         print('Error!')
#         errors = errors + 1
#     elif label == 'Young' and (i.split('/')[2] == '1-10' or i.split('/')[2] == '11-20'):
#         print('Correct!')
#         correct = correct + 1
#     elif label == 'Adult' and (i.split('/')[2] == '21-30' or i.split('/')[2] == '31-40'):
#         print('Correct!')
#         correct = correct + 1
#     elif label == 'MiddleAge' and (i.split('/')[2] == '41-50' or i.split('/')[2] == '51-60'):
#         print('Correct!')
#         correct = correct + 1
#     elif label == 'Old' and (i.split('/')[2] == '61-70' or i.split('/')[2] == '71-80' or i.split('/')[2] == '81-116'):
#         print('Correct!')
#         correct = correct + 1
#     else:
#         print('Wrong!')
#         incorrect = incorrect + 1
#
# print('CORRECT:', correct)
# print('INCORRECT:', incorrect)

