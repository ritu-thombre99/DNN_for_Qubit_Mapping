import pandas as pd
import numpy as np
from Customize_OneHot import *
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras.backend as K
from keras.layers import Input, Dense, Reshape, GlobalMaxPooling1D, Concatenate, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.models import model_from_json
from Dataset_functions import *

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

def clear_dataset(df):
    df = df.drop_duplicates()
    for i in range(5):
        df = df.drop('measure_' + str(i), axis=1)
    # Histo_Dataset(df, N_qubits=5)
    #Rimozione Features non interessanti
    df = df.drop('N_measure', axis = 1)
    df = df.drop('edge_error_02', axis=1)
    df = df.drop('edge_length_02', axis=1)
    df = df.drop('edge_error_03', axis=1)
    df = df.drop('edge_length_03', axis=1)
    df = df.drop('edge_error_04', axis=1)
    df = df.drop('edge_length_04', axis=1)
    df = df.drop('edge_error_14', axis=1)
    df = df.drop('edge_length_14', axis=1)
    df = df.drop('edge_error_20', axis=1)
    df = df.drop('edge_length_20', axis=1)
    df = df.drop('edge_error_23', axis=1)
    df = df.drop('edge_length_23', axis=1)
    df = df.drop('edge_error_24', axis=1)
    df = df.drop('edge_length_24', axis=1)
    df = df.drop('edge_error_30', axis=1)
    df = df.drop('edge_length_30', axis=1)
    df = df.drop('edge_error_32', axis=1)
    df = df.drop('edge_length_32', axis=1)
    df = df.drop('edge_error_40', axis=1)
    df = df.drop('edge_length_40', axis=1)
    df = df.drop('edge_error_41', axis=1)
    df = df.drop('edge_length_41', axis=1)
    df = df.drop('edge_error_42', axis=1)
    df = df.drop('edge_length_42', axis=1)
    return df

df = pd.read_csv('/home/ritu/dataset/dataset_tesi/NN1_Dataset(<=10Cx)_balanced1.csv')
df = clear_dataset(df)



X = df.iloc[:, 3:56].values
y = df.iloc[:, 56:].values


#MS = MinMaxScaler()
#X_st = MS.fit_transform(X)

SS = StandardScaler()
X_st = SS.fit_transform(X)

#Building validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size=0.10, random_state=1)

y_layout_train = y_train
y_layout_test = y_test

y_train = building_label(y_train)
y_test = building_label(y_test)


# load json and create model
json_file = open('/home/ritu/Tesi/Project/5qBurlington/models/NN_5Q_Balanced1_drop4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#plot_model(loaded_model, to_file='/home/rob/Scrivania/Tesi/Files/5qBurlington/NN_ReteBuona.png', show_shapes=True)

# load weights into new model
loaded_model.load_weights("/home/ritu/Tesi/Project/5qBurlington/models/NN_5Q_Balanced1_drop4.h5")
print("Loaded model from disk")
adam_optimizer = keras.optimizers.adam(learning_rate=0.0005)
loaded_model.compile(loss={'slot0':'categorical_crossentropy','slot1':'categorical_crossentropy','slot2':'categorical_crossentropy',
                     'slot3':'categorical_crossentropy','slot4':'categorical_crossentropy'},
               optimizer=adam_optimizer, metrics=['accuracy'])


# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data con metodo artigianale')
y_pred_train = loaded_model.predict(X_train)
y_pred_test = loaded_model.predict(X_test)
#results = loaded_model.evaluate(X_train,y_train)
for prediction in y_pred_train:
    print(len(prediction[0]))


def pred_layout(l):
    layout=[]
    for i in range(len(l[0])):
        layout_i = []
        for slots in l[:5]:
            if np.argmax(slots[i]) != 5:
                layout_i.append(np.argmax(slots[i]))
            else:
                layout_i.append(np.nan)
        layout.append(layout_i)
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
            for slots in l[:5]:
                if np.argmax(slots[i]) != 5:
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




def confusion_matrix(n_slot):
    #Matrice di confusione
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    y_pred = layout_test_pred[:,n_slot]
    confmat = confusion_matrix(y_true=y_layout_test[:, n_slot], y_pred=y_pred)
    print(confmat)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    tick = [0] + [0,1,2,3,4]
    print(tick)
    ax.set_xticklabels(tick)
    ax.set_yticklabels(tick)

    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('slot'+str(n_slot)+' drop4_model')
    plt.show()

for i in range(5):
    confusion_matrix(i)