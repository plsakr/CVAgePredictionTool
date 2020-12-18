import numpy as np 
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from functools import partial
import time



class ann:
    def __init__(self, nb_of_outputs, weights, X_train, X_test, y_train, y_test, entropy, nb_of_epochs):
        self.nb_of_outputs = nb_of_outputs
        self.weights = weights
        self.X_train = X_train 
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.entropy = entropy
        self.nb_of_epochs = nb_of_epochs
    
    def modelTrain(self, nb_of_outputs, weights, X_train, X_test, y_train, y_test, entropy, nb_of_epochs, save_name):
        RegularizedDense = partial(keras.layers.Dense, activation="elu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))
        model = Sequential()
        model.add(Reshape(input_shape=(1,200,200),target_shape=(200,200,1)))

        model.add(Conv2D(128, 1, activation='sigmoid'))
        model.add(MaxPool2D(pool_size=(1,1)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, 1, strides=(2,2), activation='sigmoid'))
        model.add(MaxPool2D(pool_size=(1,1)))
        model.add(BatchNormalization())

        model.add(Conv2D(32, 1, strides=(2,2), activation='sigmoid'))
        model.add(MaxPool2D(pool_size=(1,1)))
        model.add(BatchNormalization())

        model.add(Conv2D(16, 1, strides=(2,2), activation='sigmoid'))
        model.add(MaxPool2D(pool_size=(1,1)))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(RegularizedDense(48, activation='sigmoid'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(RegularizedDense(64, activation='sigmoid'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(RegularizedDense(24, activation='sigmoid'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Dense(nb_of_outputs, activation='softmax'))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_steps=10000,decay_rate=0.9)
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer, loss=entropy, metrics=['accuracy'])
        print(model.summary())
        tb = TensorBoard('./logs')
        checkpt = ModelCheckpoint(save_name,'val_accuracy',1,True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, )

        time_to_train = time.time()
        hist = model.fit(X_train,y_train,epochs=nb_of_epochs,validation_data=(X_test, y_test), callbacks=[tb, checkpt, es], class_weight=weights)

        model = load_model(save_name)
        final_predicts = model.predict(X_test)
        time_diff = time.time()
        time_train = -time_to_train + time_diff

        exact_acc = np.count_nonzero(np.absolute((y_test.argmax(axis=-1)-final_predicts.argmax(axis=-1))) <= 0) / len(y_test)
        one_off = np.count_nonzero(np.absolute((y_test.argmax(axis=-1)-final_predicts.argmax(axis=-1))) <= 1) / len(y_test)
        two_off = np.count_nonzero(np.absolute((y_test.argmax(axis=-1)-final_predicts.argmax(axis=-1))) <= 2) / len(y_test)
        print("Training time {}".format(time_train))
        print("Exact_acc = {}".format(exact_acc))
        print("One off acc = {}".format(one_off))
        print("Two off acc = {}".format(two_off))
        return exact_acc, one_off, two_off, time_train, hist

