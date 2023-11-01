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

def pick_label(circ, backend, coupling_map, optimization_level, show=False):
    '''
    Function that returns the dictionary with the mapping, choosing as the label layout the one that minimizes the depth
    of the circuit between dense layout and noise_adaptive added to the depth of operations after routing.
    In this way I also take into account which layout allows swap operations to be minimized
    '''
    new_circ_lv3 = transpile(circ, backend=backend, optimization_level=optimization_level)
    new_circ_lv3_na = transpile(circ, backend=backend, optimization_level=optimization_level,layout_method='noise_adaptive')
    #plot_circuit_layout(new_circ_lv3_na, backend).show()
    #plot_circuit_layout(new_circ_lv3, backend).show()
    cp = CouplingMap(couplinglist=coupling_map)
    depths = []
    for qc in [new_circ_lv3_na, new_circ_lv3]:
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
        if show == True:
            plot_circuit_layout(new_circ_lv3_na, backend).show()
        # qbit_mapping = (qiskit.transpiler.Layout(input_dict=new_circ_lv3_na._layout.input_qubit_mapping).get_virtual_bits())
        # return qbit_mapping
        print(new_circ_lv3_na._layout.get_physical_bits())
        return new_circ_lv3_na._layout.get_physical_bits()


    # if depths.index(min(depths)) >= 2:
    else:
        print('not na')
        if show == True:
            plot_circuit_layout(new_circ_lv3, backend).show()
        # qbit_mapping = (qiskit.transpiler.Layout(input_dict=new_circ_lv3._layout.input_qubit_mapping).get_virtual_bits())
        # return qbit_mapping
        print(new_circ_lv3._layout.get_physical_bits())
        return new_circ_lv3._layout.get_physical_bits()
        



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

    coupling_map = IBMQBackend.configuration(backend).to_dict()['coupling_map']

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


import datetime
iteration = [1]
for it in range(iteration[0]):
    # month = random.randint(1,5)
    # day = random.randint(1,29)
    # data = datetime.date(2023, month, day)
    data = datetime.datetime.today() - datetime.timedelta(days=random.randint(1,150))
    n_qs = [2,3,4,5]
    #backend_name = ['ibmq_vigo', 'ibmq_ourense',  'ibmq_rome', 'ibmq_essex','ibmq_burlington']
    backend_name_1 = ['ibm_lagos','ibm_perth','ibm_nairobi']
    # file_name = '/home/ritu/DNN_for_Qubit_Mapping/dataset/dataset_tesi/Dataset_Prova_4_08.csv'
    file_name = '/home/ritu/DNN_for_Qubit_Mapping/dataset/dataset_tesi/NN1_Dataset(<=10Cx)_balanced1.csv'
    print(data)

    for backend in backend_name_1:
        for n_q in n_qs:
            update_csv(file_name, backend, rows_to_add=1, random_n_qubit=7, random_depth=2, min_n_qubit=7, datatime=data, show=True)