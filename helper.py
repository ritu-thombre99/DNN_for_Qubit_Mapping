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
# from tensorflow.keras.layers import Conv2D, MaxPooling2D38

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from Circuit_features import *
from Backend_features import *
import qiskit
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import Unroller
from qiskit.circuit.random import random_circuit
import pandas as pd
import random
import os.path
from qiskit.transpiler.passes import LookaheadSwap, StochasticSwap
from qiskit.transpiler import CouplingMap

from qiskit.visualization import plot_circuit_layout
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends(simulator=False)

backend_dict = {}
# backend_names = ['ibm_brisbane','ibm_lagos','ibm_nairobi','ibm_perth']
# backend_names = ['ibm_lagos','ibm_nairobi','ibm_perth']
backend_names = ['ibm_brisbane','ibm_kyoto']
for backend_name in backend_names:
    backend = provider.get_backend(backend_name)
    coupling_map = IBMQBackend.configuration(backend).to_dict()['coupling_map']
    backend_dict[backend_name] = coupling_map
print(backend_dict)

num_qubits = 7
def clear_dataset(df, n_qubits):
    df = df.drop_duplicates()
    for i in range(n_qubits):
        df = df.drop('measure_' + str(i), axis=1)
    #Rimozione Features non interessanti
    df = df.drop('N_measure', axis = 1)
    # remove edges which are not in coupling maps
    # connections = ['01','10','12','21','13','31','35','53','45','54','56','65']
    connections = []
    
    if n_qubits == 7:
        for tup in backend_dict['ibm_nairobi']:
            connections.append(str(tup[0])+str(tup[1]))
    else:
        for tup in backend_dict['ibm_brisbane']:
            connections.append(str(tup[0])+str(tup[1]))
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

def get_graph_features(input_df):
    X = []
    for i in tqdm(range(len(input_df))):
        edge_features = []
        for edge_i in range(num_qubits):
            for edge_j in range(num_qubits):
                if edge_i != edge_j:
                    cx = input_df["cx_"+str(edge_i)+str(edge_j)][i]
                    edge_error, edge_length  = 100000, 100000
                    if "edge_error_"+str(edge_i)+str(edge_j) in input_df.columns:
                        edge_error = input_df["edge_error_"+str(edge_i)+str(edge_j)][i]
                        edge_length = input_df["edge_length_"+str(edge_i)+str(edge_j)][i]
                    
                    t1_i = input_df["T1_"+str(edge_i)][i]
                    t2_i = input_df["T2_"+str(edge_i)][i]
                    readout_i = input_df["readout_error_"+str(edge_i)][i]
                
                    t1_j = input_df["T1_"+str(edge_j)][i]
                    t2_j = input_df["T2_"+str(edge_j)][i]
                    readout_j = input_df["readout_error_"+str(edge_j)][i]
                    
                    
                    edge_features.append(np.array([t1_i,t2_i,readout_i,t1_j,t2_j,readout_j,cx,edge_error,edge_length]))
        X.append(np.array(edge_features).flatten())
    X = np.array(X)
    return X




# dataset functions
def pick_label(circ, backend, coupling_map, optimization_level, show=False):
    '''
    Function that returns the dictionary with the mapping, choosing as the label layout the one that minimizes the depth
    of the circuit between dense layout and noise_adaptive added to the depth of operations after routing.
    In this way I also take into account which layout allows swap operations to be minimized
    '''
    new_circ_lv3 = transpile(circ, backend=backend, optimization_level=optimization_level)
    new_circ_lv3_na = transpile(circ, backend=backend, optimization_level=optimization_level,layout_method='noise_adaptive')
    new_circ_lv3_sabre = transpile(circ, backend=backend, optimization_level=optimization_level,layout_method='sabre',routing_method='sabre')
    #plot_circuit_layout(new_circ_lv3_na, backend).show()
    #plot_circuit_layout(new_circ_lv3, backend).show()
    cp = CouplingMap(couplinglist=coupling_map)
    depths = []
    for qc in [new_circ_lv3_na, new_circ_lv3,new_circ_lv3_sabre]:
        depth = qc.depth()
        pass_manager = PassManager(LookaheadSwap(coupling_map=cp))
        lc_qc = pass_manager.run(qc)
        pass_manager = PassManager(StochasticSwap(coupling_map=cp))
        st_qc = pass_manager.run(qc)
        depths.append(depth + lc_qc.depth())
        depths.append(depth + st_qc.depth())
        #print('depth=', depth, ' depth + routing_lc_qc= ', depth + lc_qc.depth(), ' depth + routing_st_qc=',depth + st_qc.depth())
    print("Depth:",depths)
    if depths.index(min(depths)) < 2:
        print('na')
        # if show == True:
        #     plot_circuit_layout(new_circ_lv3_na, backend).show()
        return new_circ_lv3_na._layout.get_physical_bits()

    if depths.index(min(depths)) >= 2 and depths.index(min(depths)) <4:
        print('not na')
        # if show == True:
        #     plot_circuit_layout(new_circ_lv3, backend).show()
        return new_circ_lv3._layout.get_physical_bits()

    else:
        print('SABRE')
        # if show == True:
        #     plot_circuit_layout(new_circ_lv3_sabre, backend).show()
        return new_circ_lv3_sabre._layout.get_physical_bits()
        



def add_line(circuit, backend_name, refresh=True, show=True, optimization_level=3, datatime=False):
    '''
    Function that constructs rows of the dataset returning me the one made from a first component which is
    a list containing the titles of the features, and a second list containing the corresponding values.
    _____________________________________________________________________________________________________________________
    The label titles are the last features of the row: the title is the name of the physical slot, the value
    is the name of the virtual qubit
    '''

    provider = IBMQ.get_provider(hub='ibm-q')
    provider.backends(simulator=False)
    backend = provider.get_backend(backend_name)
    basis_gats = backend.configuration().basis_gates
    # print(basis_gats)
    pass_ = Unroller(basis_gats)
    pm = PassManager(pass_)

    # new_circ = pm.run(circuit)
    # need to transpile before passing to run
    new_circ = transpile(circuit, backend=backend, optimization_level=0)
    new_circ = pm.run(new_circ)

    size_backend = len(backend.properties(refresh=refresh).to_dict()['qubits'])

    CA = circuit_analysis(backend, circuit, size_backend=size_backend, show=show)
    #print(CA)
    print(datatime)
    BT = Backend_Topology(backend_name, refresh, show, datatime=datatime)
    #print(BT)
    QP = Qubit_properties(backend_name, refresh, datatime=datatime)
    #print(QP)
    # coupling_map = IBMQBackend.configuration(backend).to_dict()['coupling_map']
    coupling_map = backend_dict[backend_name]
    

    label = pick_label(new_circ, backend=backend, coupling_map=coupling_map, optimization_level=optimization_level, show = show)
    #print(label)

    Title_names =['last_update_date', 'backend_name'] + list(CA.keys())[:3] + list(CA['cx'].keys()) + list(CA['measure'].keys()) + list(BT['coupling'].keys())
    for i in range(size_backend):
        for title in list(QP.keys())[2:]:
            Title_names.append(title+'_'+str(i))

    Title_names = Title_names + list(range(size_backend))


    date_name = [BT['last_update_date'], BT['backend_name']]
    Values = date_name + list(CA.values())[:3] + list(CA['cx'].values()) + list(CA['measure'].values()) + list(BT['coupling'].values())
    for i in range(size_backend):
        for value in list(QP.keys())[2:]:
            Values.append(QP[value][i])

    for qubit in range(size_backend):
        #print(label[qubit].register.name)
        if label[qubit].register.name == 'q':
            Values.append(label[qubit].index)
        else:
            Values.append(None)



    return [Title_names,Values]




def update_csv(file_name, backend_name, rows_to_add, random_n_qubit=5, random_depth=10, show = False, min_n_qubit=1, datatime=False):

    '''
    Function that adds the row to the dataset contained in file_name.
    __________________________________________________________________________________
    backend_name = backend on which to map
    rows_to_add = number of lines to add varies randomly the number of qubits and the depth of the circuit
    random_n_qubit = upper bound of randint (default 5)
    random_depth = upper bound of randint for depth (default 10)
    '''
    if os.path.exists(file_name) == True:
        df = pd.read_csv(file_name)
        # print(df.columns)
        # start = len(list(df['Index']))
        start = len(list(df['Unnamed: 0']))
        

    else:
        start = 0
    print(start)
    for j in range(start, start + rows_to_add):
        # print(j, backend_name, j-start+1)
        n_qubit = random.randint(min_n_qubit, random_n_qubit)
        depth = random.randint(1, random_depth)
        try:
            circ = random_circuit(n_qubit, depth, measure=True)
            #circ.draw(output='mpl').show()
            l = add_line(circ,backend_name, optimization_level=3, refresh=True, show= show, datatime=datatime)
        except qiskit.transpiler.exceptions.TranspilerError :
            print('Unable to map the circuit: Generating new sth')
            error = 1
            while error==1:
                try:
                    circ = random_circuit(n_qubit, depth, measure=True)
                    # circ.draw(output='mpl').show()
                    l = add_line(circ, backend_name, optimization_level=3, refresh=True, show=show)
                    error = 0
                except qiskit.transpiler.exceptions.TranspilerError:
                    continue

        d={}

        for i in range(len(l[0])):
            d[str(l[0][i])] = l[1][i]

        #print(d)

        df = pd.DataFrame(d, index=[j])
        if j==0:
            df.to_csv(file_name, mode='a')
        else:
            df.to_csv(file_name, mode='a',header=None)


# dnn pred layout functions

def pred_layout(l, num_qubits):
    layout=[]
    for i in range(len(l[0])):
        layout_i = []
        for slots in l[:num_qubits]:
            if np.argmax(slots[i]) != num_qubits:
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

def pred_layout_diff_elem(l,num_qubits):
    layout=[]
    for i in range(len(l[0])):
        check = 0
        while check != 1:
            layout_i = []
            for slots in l[:num_qubits]:
                if np.argmax(slots[i]) != num_qubits:
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

def get_labels(y):
    labels = []
    for i in range(len(y)):
        labels.append(np.where(y[i]==np.max(y[i]))[0][0])
    return labels

# get times to generate dataframes
# def get_df_time(circuit,backend_name,datatime,show=False,refresh=True):
#     start = time.time()
#     provider = IBMQ.get_provider(hub='ibm-q')
#     provider.backends(simulator=False)
#     backend = provider.get_backend(backend_name)
#     basis_gats = backend.configuration().basis_gates
#     # print(basis_gats)
#     pass_ = Unroller(basis_gats)
#     pm = PassManager(pass_)

#     # new_circ = pm.run(circuit)
#     # need to transpile before passing to run
#     new_circ = transpile(circuit, backend=backend, optimization_level=0)
#     new_circ = pm.run(new_circ)
    
#     size_backend = len(backend.properties(refresh=refresh).to_dict()['qubits'])
#     CA = circuit_analysis(backend, circuit, size_backend=size_backend, show=show)
#     BT = Backend_Topology(backend_name, refresh, show, datatime=datatime)
#     QP = Qubit_properties(backend_name, refresh, datatime=datatime)
    
#     Title_names =['last_update_date', 'backend_name'] + list(CA.keys())[:3] + list(CA['cx'].keys()) + list(CA['measure'].keys()) + list(BT['coupling'].keys())
#     for i in range(size_backend):
#         for title in list(QP.keys())[2:]:
#             Title_names.append(title+'_'+str(i))

#     date_name = [BT['last_update_date'], BT['backend_name']]
#     Values = date_name + list(CA.values())[:3] + list(CA['cx'].values()) + list(CA['measure'].values()) + list(BT['coupling'].values())
#     for i in range(size_backend):
#         for value in list(QP.keys())[2:]:
#             Values.append(QP[value][i])
#     df = pd.DataFrame([Values],columns = Title_names)
#     df = clear_dataset(df, 7)
#     end = time.time()
#     return (end-start)
