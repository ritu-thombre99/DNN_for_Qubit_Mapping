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


import datetime
from helper import *


def main():
    iteration = [1000]
    for it in range(iteration[0]):
        # month = random.randint(1,5)
        # day = random.randint(1,29)
        # data = datetime.date(2023, month, day)
        data = datetime.today() - timedelta(days=random.randint(1,150))
        n_qs = [2,3,4,5]
        backend_name_1 = [('ibm_brisbane',127)]
        # backend_name_1 = [('ibm_lagos',7),('ibm_perth',7),('ibm_nairobi',7)]
        # file_name = '/home/ritu/DNN_for_Qubit_Mapping/dataset/dataset_tesi/Dataset_Prova_4_08.csv'
        file_name = 'dataset/dataset_tesi/NN1_Dataset(<=10Cx)_balanced1.csv'
        print(data)
        
        # for n_q in n_qs:
        for _ in range(2):
            for backend in backend_name_1:
                backend_name, n_qubits = backend
                if backend_name != 'ibm_brisbane':
                    update_csv(file_name, backend_name, rows_to_add=1, random_n_qubit=n_qubits, random_depth=2, min_n_qubit=n_qubits//2, datatime=data, show=True)
                else:
                    update_csv(file_name, backend_name, rows_to_add=1, random_n_qubit=20, random_depth=2, min_n_qubit=9, datatime=data, show=True)
                    




if __name__ == "__main__":
    main()