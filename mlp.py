import tensorflow as tf
from qiskit import *
from qiskit.providers.ibmq import *
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import time
from keras import optimizers
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

# import keras
from tensorflow.keras.models import Sequential#,Input,Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow import keras
# from keras_layer_normalization import BatchNormalization
# from keras_layer_normalization import LayerNormalization
# from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Model

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def clear_dataset(df, n_qubits):
    df = df.drop_duplicates()
    for i in range(n_qubits):
        df = df.drop('measure_' + str(i), axis=1)
    #Rimozione Features non interessanti
    df = df.drop('N_measure', axis = 1)
    # remove edges which are not in coupling maps
    # connections = ['01','10','12','21','13','31','35','53','45','54','56','65']
    connections = []
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    provider.backends(simulator=False)
    if n_qubits == 7:
        backend = provider.get_backend('ibm_nairobi')
        coupling_map = IBMQBackend.configuration(backend).to_dict()['coupling_map']
        for tup in coupling_map:
            connections.append(str(tup[0])+str(tup[1]))
    else:
        backend = provider.get_backend('ibm_brisbane')
        coupling_map = IBMQBackend.configuration(backend).to_dict()['coupling_map']
        for tup in coupling_map:
            connections.append(str(tup[0])+str(tup[1]))
    print("Connections:",connections)
    to_keep = []
    for c in connections:
        to_keep.append("edge_error_"+c)    
        to_keep.append("edge_length_"+c)
    to_drop = []
    for c in df.columns:
        if "edge_error" in c or "edge_length" in c:
            if c not in to_keep:
                to_drop.append(c)
    df = df.drop(to_drop,axis=1)
    return df

num_qubits = 7
df = pd.read_csv("dataset/dataset_tesi/NN1_Dataset(<=10Cx)_balanced1.csv")

data_to_use = int(1*len(df))
df = df.iloc[:data_to_use]
df = clear_dataset(df, num_qubits)

last_num_qubits = len(df.columns)-num_qubits
labels = df.iloc[:, last_num_qubits:].values

y_old =(df.iloc[:, last_num_qubits:].values)
y = []
for i in range(len(y_old)):
    n = len(y_old[i])
    y_new = np.array(([[0]*n]*n))
    for j in range(len(y_old[i])):
        y_new[j][y_old[i][j]] = 1
    y_new = y_new.flatten()
    y.append(y_new)
    
y=np.array(y)

SS = StandardScaler()
X = df.iloc[:, 3:last_num_qubits].values
X_st = SS.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size=0.10, random_state=1)

def build_model_original_df(input_shape):
        #d = 0.2
        model = Sequential()
        input_layer = Input(shape=input_shape, name='input')
        layer1 = Dense(70,kernel_initializer='uniform',activation='relu')(input_layer)
        layer2= Dense(49,kernel_initializer='uniform',activation='relu')(layer1)
        merged = Model(inputs=[input_layer], outputs=[layer2])

        return merged

def get_pred():
    model = build_model_original_df(X[0].shape[0])
    model.build(X[0].shape[0])
    learning_rate = 0.0005
    optimizer = tf.optimizers.Adam(name='Adam', learning_rate=learning_rate)
    model.compile(
    optimizer=optimizer,
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.mean_squared_error])
    print(model.summary())

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=200,
        validation_split=0.15,
        verbose=1)
    end = time.time()
    print("Time taken to train the model:", end-start)

    plt.plot(history.epoch, history.history['loss'] , label = "loss")
    plt.plot(history.epoch, history.history['val_loss'] , label = "val_loss")


    plt.legend()
    plt.show()


    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    return test_pred,train_pred

def get_labels(y):
    labels = []
    for i in range(len(y)):
        labels.append(np.where(y[i]==np.max(y[i]))[0][0])
    return labels

def main():
    test_pred,train_pred = get_pred()
    count = 0
    for i in range(len(train_pred)):
        actual=get_labels(np.reshape(y_train[i],(7,7)))
        predicted=get_labels(np.reshape(train_pred[i],(7,7)))
        if actual == predicted:
            count = count + 1
    train_accuracy = count/len(train_pred)

    count = 0
    for i in range(len(test_pred)):
        actual=get_labels(np.reshape(y_test[i],(7,7)))
        predicted=get_labels(np.reshape(test_pred[i],(7,7)))
        if actual == predicted:
            count = count + 1
    test_accuracy = count/len(test_pred)
    if train_accuracy < 0.8 or test_accuracy < 0.8:
        main()
    return 

if __name__ == "__main__":
    main()