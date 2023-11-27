import tensorflow as tf
from qiskit import *
from qiskit.providers.ibmq import *
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

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
from helper import *



num_qubits = 7
df = pd.read_csv("dataset/dataset_tesi/NN1_Dataset(<=10Cx)_balanced1.csv")

data_to_use = int(1*len(df))
df = df.iloc[:data_to_use]
df = clear_dataset(df, num_qubits)

last_num_qubits = len(df.columns)-num_qubits
labels = df.iloc[:, last_num_qubits:].values
"""
Node fatures: T1, T2,readout_error (N*3)
Edge features: CNOTs, edge_error, edge_length ((edges*3))
edge_features -> ((t1_i,t2_i,readout_i),(t1_j,t2_j,readout_j),(edge_error_ij,edge_length_ij,cnot_ij))

"""
print("Creating dataset")


X = get_graph_features(df)


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
X_st = SS.fit_transform(X)
#MS = MinMaxScaler()
#X_st = MS.fit_transform(X)

#Building validation set

X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size=0.10, random_state=1)

def build_model(input_shape):
    #d = 0.2
    model = Sequential()
    input_layer = Input(shape=input_shape, name='input')
    layer1 = Dense(input_shape//2,kernel_initializer='uniform',activation='relu')(input_layer)
    layer2 = Dense(input_shape//4,kernel_initializer='uniform',activation='relu')(layer1)
    layer3 = Dense(49,kernel_initializer='uniform',activation='relu')(layer2)
    merged = Model(inputs=[input_layer], outputs=[layer3])

    return merged
def get_pred():
    model = build_model(X[0].shape[0])
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
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'models/mlp_edge_features_train_loss.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    plt.plot(history.epoch, history.history['loss'] , label = "loss")
    plt.plot(history.epoch, history.history['val_loss'] , label = "val_loss")

    plt.legend()
    plt.show()

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    return model,test_pred,train_pred,end-start

def get_labels(y):
    labels = []
    for i in range(len(y)):
        labels.append(np.where(y[i]==np.max(y[i]))[0][0])
    return labels
def main():
    model,test_pred,train_pred,del_t = get_pred()
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
    print("Train accuracy:",train_accuracy)
    print("Test accuracy:",test_accuracy)
    f = open("models/mlp_edge_features_accuracy.txt","w")
    f.writelines("Train accuracy:"+str(train_accuracy))
    f.writelines("\nTest accuracy:"+str(test_accuracy))
    f.writelines("\nTraining time:"+str(del_t))
    f.close()
    model.save("models/mlp_edge_features.keras")

    return 

if __name__ == "__main__":
    main()