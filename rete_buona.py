import pandas as pd
import numpy as np
from Customize_OneHot import *
from Dataset_functions import *
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import plot_model


np.random.seed(123)
tf.random.set_seed(123)




def accuracy(y_pred, y_train):
    count = 0
    for i in range(y_train.shape[0]):
       if nan_equal(y_pred[i], y_train[i]):
           count = count+1
    return count/y_train.shape[0]


def building_label(y):
    y_label = []
    for row in range(y.shape[0]):
        label = customize_OH_withNan(list(y[row]))
        y_label.append(label)
    y_label = np.array(y_label)
    return y_label

def nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True

def clear_dataset(df, n_qubits):
    df = df.drop_duplicates()
    for i in range(n_qubits):
        df = df.drop('measure_' + str(i), axis=1)
    # Histo_Dataset(df, N_qubits=n_qubits)
    #Rimozione Features non interessanti
    df = df.drop('N_measure', axis = 1)
    # TO DO
    # remove edges which are not in coupling maps
    connections = ['01','10','12','21','13','31','35','53','45','54','56','65']
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

    
    # df = df.drop('edge_error_02', axis=1)
    # df = df.drop('edge_length_02', axis=1)
    # df = df.drop('edge_error_03', axis=1)
    # df = df.drop('edge_length_03', axis=1)
    # df = df.drop('edge_error_04', axis=1)
    # df = df.drop('edge_length_04', axis=1)
    # df = df.drop('edge_error_14', axis=1)
    # df = df.drop('edge_length_14', axis=1)
    # df = df.drop('edge_error_20', axis=1)
    # df = df.drop('edge_length_20', axis=1)
    # df = df.drop('edge_error_23', axis=1)
    # df = df.drop('edge_length_23', axis=1)
    # df = df.drop('edge_error_24', axis=1)
    # df = df.drop('edge_length_24', axis=1)
    # df = df.drop('edge_error_30', axis=1)
    # df = df.drop('edge_length_30', axis=1)
    # df = df.drop('edge_error_32', axis=1)
    # df = df.drop('edge_length_32', axis=1)
    # df = df.drop('edge_error_40', axis=1)
    # df = df.drop('edge_length_40', axis=1)
    # df = df.drop('edge_error_41', axis=1)
    # df = df.drop('edge_length_41', axis=1)
    # df = df.drop('edge_error_42', axis=1)
    # df = df.drop('edge_length_42', axis=1)
    return df

df = pd.read_csv('dataset/dataset_tesi/NN1_Dataset(<=10Cx)_balanced1.csv')
df = clear_dataset(df, 7)


last_7 = len(df.columns)-7
X = df.iloc[:, 3:last_7].values
y = df.iloc[:, last_7:].values
print("Lengths:",len(X),len(y))
''''
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if X[i, j] == 100000:
            X[i, j] = -10

for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        if y[i, j] == 10:
            y[i, j] = np.nan
'''


SS = StandardScaler()
X_st = SS.fit_transform(X)
#MS = MinMaxScaler()
#X_st = MS.fit_transform(X)

#Building validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size=0.10, random_state=1)

from sklearn.utils import class_weight


class_weight_0 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:,0]), y=y_train[:,0])
# class_weight_0 = dict(enumerate(class_weight_0))

class_weight_1 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:,1]), y=y_train[:,1])
# class_weight_1 = dict(enumerate(class_weight_1))

class_weight_2 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:,2]), y=y_train[:,2])
# class_weight_2 = dict(enumerate(class_weight_2))

class_weight_3 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:,3]), y=y_train[:,3])
# class_weight_3 = dict(enumerate(class_weight_3))

class_weight_4 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:,4]), y=y_train[:,4])

class_weight_5 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:,5]), y=y_train[:,5])

class_weight_6 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train[:,6]), y=y_train[:,6])

print(class_weight_0, class_weight_1, class_weight_2, class_weight_3, class_weight_4, class_weight_5, class_weight_6 )

y_layout_train = y_train
y_layout_test = y_test

y_train = building_label(y_train)
y_test = building_label(y_test)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

A0 = Input(shape=(X_train.shape[1], ), name='input')
A1 = Dense(264, activation='relu')(A0)
A2 = Dense(1024, activation='relu',kernel_regularizer=keras.regularizers.l1(0.001))(A1)
drop = Dropout(0.65)(A2)

#Example of 'Slot Structure'
A3_0 = Dense(256, activation='relu')(drop)
A4_0 = Dense(128, activation='relu')(A3_0)
slot0 = Dense(units=6, activation='softmax', name='slot0')(A4_0)

A3_1 = Dense(256, activation='relu')(drop)
A4_1 = Dense(128, activation='relu')(A3_1)
slot1 = Dense(units=6, activation='softmax', name='slot1')(A4_1)


A3_2 = Dense(256, activation='relu')(drop)
A4_2 = Dense(128, activation='relu')(A3_2)
slot2 = Dense(units=6, activation='softmax', name='slot2')(A4_2)


A3_3 = Dense(256, activation='relu')(drop)
A4_3 = Dense(128, activation='relu')(A3_3)
slot3 = Dense(units=6, activation='softmax', name='slot3')(A4_3)

A3_4 = Dense(256, activation='relu')(drop)
drop_4 = Dropout(0.65)(A3_4)
A4_4 = Dense(128, activation='relu')(drop_4)
slot4 = Dense(units=6, activation='softmax', name='slot4')(A4_4) 

A3_5 = Dense(256, activation='relu')(drop)
A4_5 = Dense(128, activation='relu')(A3_5)
slot5 = Dense(units=6, activation='softmax', name='slot5')(A4_5)

A3_6 = Dense(256, activation='relu')(drop)
A4_6 = Dense(128, activation='relu')(A3_6)
slot6 = Dense(units=6, activation='softmax', name='slot6')(A4_6)



merged = Model(inputs=[A0], outputs=[slot0, slot1, slot2, slot3, slot4,slot5,slot6])
print(merged.summary())
#plot_model(merged, to_file='/home/ritu/Tesi/Project/5qBurlington/NN2_b.png', show_shapes=True)

# adam_optimizer = keras.optimizers.adam(learning_rate=0.0005)
adam_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
merged.compile(loss={'slot0':'categorical_crossentropy','slot1':'categorical_crossentropy','slot2':'categorical_crossentropy',
                     'slot3':'categorical_crossentropy','slot4':'categorical_crossentropy','slot5':'categorical_crossentropy','slot6':'categorical_crossentropy'},
               optimizer=adam_optimizer, metrics=['accuracy'])


# history = merged.fit({'input': X_train},  {'slot0': y_train[:,0:6],'slot1':y_train[:,6:12],'slot2':y_train[:,12:18],'slot3':y_train[:,18:24],
#                               'slot4':y_train[:,24:30]},
#            epochs=175, batch_size=128, verbose=1, validation_split=0.10, shuffle=True,
#                      class_weight={'slot0':class_weight_0, 'slot1':class_weight_1, 'slot2':class_weight_2,
#                                      'slot3':class_weight_3, 'slot4':class_weight_4})


history = merged.fit({'input': X_train},  {'slot0': y_train[:,0:6],'slot1':y_train[:,6:12],'slot2':y_train[:,12:18],'slot3':y_train[:,18:24],
                              'slot4':y_train[:,24:30],'slot5':y_train[:,30:36],'slot6':y_train[:,36:42]},
           epochs=175, batch_size=128, verbose=1, validation_split=0.10, shuffle=True,
                     class_weight={0:class_weight_0, 1:class_weight_1, 2:class_weight_2,
                                     3:class_weight_3, 4:class_weight_4, 5:class_weight_5, 6:class_weight_6})

''''
history = merged.fit({'input': X_train},  {'slot0': y_train[:,0:6],'slot1':y_train[:,6:12],'slot2':y_train[:,12:18],'slot3':y_train[:,18:24],
                              'slot4':y_train[:,24:30]},
           epochs=150, batch_size=150, verbose=1, validation_split=0.10, shuffle=True)
'''''
'''
# serialize model to JSON
model_json = merged.to_json()
with open("/home/ritu/Tesi/Project/5qBurlington/models/NN_5Q_Balanced1_drop4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
merged.save_weights("/home/ritu/Tesi/Project/5qBurlington/models/NN_5Q_Balanced1_drop4.h5")
print("Saved model to disk")
'''

print('\n# Evaluate on test data')
results = merged.evaluate(X_test, {'slot0': y_test[:,0:6],'slot1':y_test[:,6:12],'slot2':y_test[:,12:18],'slot3':y_test[:,18:24],
                                   'slot4': y_test[:,24:30],'slot5':y_test[:,30:36],'slot6':y_test[:,36:42]}, batch_size=128)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data con metodo artigianale')
y_pred_train = merged.predict(X_train)
y_pred_test = merged.predict(X_test)
#results = merged.evaluate(X_train,y_train)
for prediction in y_pred_train:
    print(len(prediction[0]))


def pred_layout(l):
    layout=[]
    for i in range(len(l[0])):
        layout_i = []
        for slots in l[:7]:
            if np.argmax(slots[i]) != 7:
                layout_i.append(np.argmax(slots[i]))
            else:
                layout_i.append(np.nan)
        layout.append(layout_i)
    print(layout)
    return layout

def controlla_rip(l):
    rip = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i] == l[j]:
                if (i,j) not in rip:
                    rip.append((i,j))

    return rip

def pred_layout_diff_elem(l):
    layout=[]
    for i in range(len(l[0])):
        check = 0
        while check != 1:
            layout_i = []
            for slots in l[:7]:
                if np.argmax(slots[i]) != 7:
                    layout_i.append(np.argmax(slots[i]))
                else:
                    layout_i.append(np.nan)
            rip = controlla_rip(layout_i)
            if rip == []:
                check = 1
            else:
                for index in rip:
                    if l[index[0]][i][layout_i[index[0]]] > l[index[1]][i][layout_i[index[1]]]:
                        l[index[1]][i][layout_i[index[1]]]=0
                    else:
                        l[index[0]][i][layout_i[index[0]]] = 0

        layout.append(layout_i)
    return layout



layout_train_pred = np.array(pred_layout(y_pred_train))
layout_test_pred = np.array(pred_layout(y_pred_test))
layout_train_pred_nr = np.array(pred_layout_diff_elem(y_pred_train))
layout_test_pred_nr = np.array(pred_layout_diff_elem(y_pred_test))

count_train = 0
count_train_nr = 0
for i in range(y_train.shape[0]):
    if nan_equal(layout_train_pred[i,:], y_layout_train[i, :]):
        count_train = count_train + 1
    if nan_equal(layout_train_pred_nr[i,:], y_layout_train[i, :]):
        count_train_nr = count_train_nr + 1
print('acc_Train', count_train/y_train.shape[0])
print('acc_Train_nr', count_train_nr/y_train.shape[0])

count_test = 0
count_test_nr = 0
for i in range(y_test.shape[0]):
    if nan_equal(layout_test_pred[i,:], y_layout_test[i, :]):
        count_test = count_test + 1
    if nan_equal(layout_test_pred_nr[i,:], y_layout_test[i, :]):
        count_test_nr = count_test_nr + 1
print('acc_Test', count_test/y_test.shape[0])
print('acc_Test_nr', count_test_nr/y_test.shape[0])


''''
output_dense_pred_train = y_pred_train[5]
output_dense_pred_test = y_pred_test[5]
train_01 = []
for i in range(len(output_dense_pred_train)):
  train_01.append(max_per_slot(output_dense_pred_train[i]))
train_01 = np.array(train_01)
test_01 = []
for i in range(len(output_dense_pred_test)):
  test_01.append(max_per_slot(output_dense_pred_test[i]))
test_01 = np.array(test_01)


print('Accuracy train', accuracy(train_01,y_train))
print('Accuracy test', accuracy(test_01,y_test))

'''''

from matplotlib import pyplot
# plot history
pyplot.plot(history.history['val_slot0_accuracy'], label='val_slot0')
pyplot.plot(history.history['val_slot1_accuracy'], label='val_slot1')
pyplot.plot(history.history['val_slot2_accuracy'], label='val_slot2')
pyplot.plot(history.history['val_slot3_accuracy'], label='val_slot3')
pyplot.plot(history.history['val_slot4_accuracy'], label='val_slot4')
#pyplot.plot(history.history['val_slot01234_accuracy'], label='slot_C01234')
pyplot.legend()
pyplot.show()
pyplot.plot(history.history['val_slot0_loss'], label='val_slot0_los')
pyplot.plot(history.history['val_slot1_loss'], label='val_slot1_los')
pyplot.plot(history.history['val_slot2_loss'], label='val_slot2_los')
pyplot.plot(history.history['val_slot3_loss'], label='val_slot3_los')
pyplot.plot(history.history['val_slot4_loss'], label='val_slot4_los')
#pyplot.plot(history.history['val_slot01234_loss'], label='slot_C01234_los')
pyplot.legend()
pyplot.show()
