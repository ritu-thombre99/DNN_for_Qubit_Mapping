import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import Unroller


def circuit_analysis(backend, circuit, size_backend, show=False):
    '''Funzione che mi ritorna il numero di qubits del circuito,
    il numero di CNot per ogni coppia e sua inversa,
    il numero di misure per ogni qubit
    ----------------------------------------------------
    se show = True mostra il circuito unrollato'''
    circuit_analysis = {}

    #Unroll circuit
    pass_ = Unroller(['u1', 'u2', 'u3', 'cx', 'id'])
    pm = PassManager(pass_)
    # new_circ = pm.run(circuit)
    # need to transpile before passing to run
    new_circ = transpile(circuit, backend=backend, optimization_level=0)
    new_circ = pm.run(new_circ)

    if show == True:
        new_circ.draw(output='mpl').show()


    dag = circuit_to_dag(new_circ)
    N_qubits = dag.properties()['qubits']
    if check_in_list('cx',list(dag.properties()['operations'].keys())) != []:
        N_cx = dag.properties()['operations']['cx']
    else:
        N_cx = 0
    if check_in_list('measure', list(dag.properties()['operations'].keys())) != []:
        N_measure = dag.properties()['operations']['measure']
    else:
        N_measure = 0

    circuit_analysis['N_qubtis'] = N_qubits
    circuit_analysis['N_cx'] = N_cx
    circuit_analysis['N_measure'] = N_measure

    CNots = CNot_count(dag)
    Measures = measure_count(dag)

    control_qubit = []
    execution_qubit = []


    for i in range(len(CNots)):
        control_qubit.append(CNots[i][1])
        execution_qubit.append(CNots[i][2])
    #print(control_qubit)
    #print(execution_qubit)

    cx_dict = {}
    measure_dict = {}
    for i in range(size_backend):

        '''Riempio Dizionario misure'''
        str_dict_mis = 'measure_' + str(i)
        measure_dict[str_dict_mis] = len(check_in_list(i,Measures))
        for j in range(size_backend):
            '''Riempio Dizionario cx'''
            if(i != j ):
                string = 'cx_' + str(i) + str(j)
                list_1 = check_in_list(i, control_qubit)
                list_2 = check_in_list(j, execution_qubit)

                count = 0
                for control_index in list_1:
                    for esecution_index in list_2:
                        if control_index == esecution_index:
                            count = count + 1
                        else:
                            count = count

                cx_dict[string] = count

    circuit_analysis['cx'] = cx_dict
    if sum(cx_dict.values()) != N_cx :
        print('N_cx is different from the cx sum' )

    circuit_analysis['measure'] = measure_dict
    if sum(measure_dict.values()) != N_measure :
        print('N_measure is different from the measure sum' )

    return circuit_analysis




def CNot_count(dag):
    '''Funzione che ritorna una lista con tutti i CNot all'interno del
    Circuito DAG. Ogni CNot Sarà a sua volta una lista con l'indice del nodo a cui esso i trova
    con qubit di controllo e con il qubit di esecuzione'''
    CNots=[]
    node_numbers = len(dag.op_nodes())
    for i in range(node_numbers):
        node = dag.op_nodes()[i]
        if node.name == 'cx':
            CNots.append([i,node.qargs[0].index,node.qargs[1].index])

    #print(CNots)
    return CNots


def measure_count(dag):
    '''Funzione che ritorna una lista con tutte le misure all'interno del
    Circuito DAG.'''
    measures=[]
    node_numbers = len(dag.op_nodes())
    for i in range(node_numbers):
        node = dag.op_nodes()[i]
        if node.name == 'measure':
            measures.append(node.qargs[0].index)

    #print(measures)
    return measures


def check_in_list(i, list):
    '''funzione che mi dice in che posizione si trovano gli elementi i nella lista list'''
    index = []
    for j in range(len(list)):
        if i == list[j]:
            index.append(j)
        else:
            continue

    return index