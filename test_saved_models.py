from get_test_datasets import return_dataset
test_dataset = return_dataset()
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helper import *

mlp_edge_feature = tf.keras.models.load_model("models/mlp_edge_features.keras")
dnn = tf.keras.models.load_model("models/dnn.keras")
mlp = tf.keras.models.load_model("models/mlp.keras")

basis_gate_dict = {}
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends(simulator=False)
backends = ['ibm_lagos','ibm_perth','ibm_nairobi']
for backend_name in backends:
    backend = provider.get_backend(backend_name)
    basis_gats = backend.configuration().basis_gates
    basis_gate_dict[backend_name] = basis_gats


def get_transpiled_circ_results(qc, backend, initial_layout):
    basis_gats = basis_gate_dict[backend]
    pass_ = Unroller(basis_gats)
    pm = PassManager(pass_)
    backend = provider.get_backend(backend)
    
    tqc = transpile(qc, backend=backend,initial_layout=initial_layout, optimization_level=0,seed_transpiler=13)
    tqc = pm.run(tqc)
    gates = dict(tqc.count_ops())
    return gates['cx'], tqc.depth()




SS = StandardScaler()
num_qubits = 7
count_or = 0
count_nr = 0
count_mlp = 0
count_mlp_edge = 0
original_depth, original_CNOT = [],[]
mlp_edge_depth, mlp_edge_CNOT = [],[]
dnn_without_ro_depth,dnn_without_ro_CNOT = [],[] 
dnn_with_ro_depth,dnn_with_ro_CNOT = [],[] 
mlp_depth, mlp_CNOT = [],[]
for qc_i in tqdm(range(len(test_dataset))):
    qc = test_dataset[qc_i]
    data = datetime.today() - timedelta(days=random.randint(1,150))
    backends = ['ibm_lagos','ibm_perth','ibm_nairobi']
    backend = backends[np.random.randint(0,3)]
    l = add_line(qc,backend, optimization_level=3, refresh=True, show= False, datatime=data)
    
    d={}
    for i in range(len(l[0])):
        d[str(l[0][i])] = l[1][i]
    df = pd.DataFrame(d, index=[0])
    df = clear_dataset(df, 7)
    last_num_qubits = len(df.columns)-num_qubits

    x = SS.fit_transform(df.iloc[:, 2:last_num_qubits].values)
    labels = ((df.iloc[:, last_num_qubits:].values)[0]).tolist()
    
    predicted = dnn.predict(x)
    
    pred_or = np.array(pred_layout(predicted, num_qubits))[0].tolist()
    pred_nr = np.array(pred_layout_diff_elem(predicted,num_qubits))[0].tolist()
    print(pred_or, pred_nr, labels)
    if(pred_or == labels):
        count_or = count_or + 1
    if(pred_nr == labels):
        count_nr = count_nr + 1
    
    mlp_pred = get_labels(np.reshape(mlp.predict(x),(7,7)))
    print(mlp_pred)
    if mlp_pred == labels:
        count_mlp = count_mlp + 1
        
        
    x = SS.fit_transform(get_graph_features(df))
    mlp_edges_pred = get_labels(np.reshape(mlp_edge_feature.predict(x),(7,7)))
    print(mlp_edges_pred)
    if mlp_edges_pred == labels:
        count_mlp_edge = count_mlp_edge + 1
    
    
    # original circuit after unrolling
    cx, depth = get_transpiled_circ_results(qc, backend, initial_layout=labels)
    original_depth.append(depth)
    original_CNOT.append(cx)
    print(original_depth[-1],original_CNOT[-1])
    if len(mlp_edges_pred) == len(set(mlp_edges_pred)):
        cx, depth = get_transpiled_circ_results(qc, backend, initial_layout=mlp_edges_pred)
        mlp_edge_depth.append(depth)
        mlp_edge_CNOT.append(cx)
    else:
        mlp_edge_depth.append(float('NaN'))
        mlp_edge_CNOT.append(float('NaN'))
        
    # NN without edge features
    if len(mlp_pred) == len(set(mlp_pred)):
        cx, depth = get_transpiled_circ_results(qc, backend, initial_layout=mlp_pred)
        mlp_depth.append(depth)
        mlp_CNOT.append(cx)
    else:
        mlp_depth.append(float('NaN'))
        mlp_CNOT.append(float('NaN'))
        
    # DNN without repair operator
    if len(pred_or) == len(set(pred_or)):
        cx, depth = get_transpiled_circ_results(qc, backend, initial_layout=pred_or)
        dnn_without_ro_depth.append(depth)
        dnn_without_ro_CNOT.append(cx)
    else:
        dnn_without_ro_depth.append(float("NaN"))
        dnn_without_ro_CNOT.append(float("NaN"))
        
    # DNN with RO
    cx, depth = get_transpiled_circ_results(qc, backend, initial_layout=pred_nr)
    dnn_with_ro_depth.append(depth)
    dnn_with_ro_CNOT.append(cx)
print("DNN accuracy without repair operator:", count_or/len(test_dataset))
print("DNN accuracy with repair operator:", count_nr/len(test_dataset))
print("MLP accuracy:", count_mlp/len(test_dataset))
print("MLP with edge features accuracy:", count_mlp_edge/len(test_dataset))

df = pd.DataFrame()
df["Label Depth"] = original_depth
df["Label CNOTs"] = original_CNOT

df["NN without edge features Depth"] = mlp_depth
df["NN without edge features CNOTs"] = mlp_CNOT

df["NN with edge features Depth"] = mlp_edge_depth
df["NN with edge features CNOTs"] = mlp_edge_CNOT

df["DNN without repair Depth"] = dnn_without_ro_depth
df["DNN without repair CNOTs"] = dnn_without_ro_CNOT

df["DNN with repair Depth"] = dnn_with_ro_depth
df["DNN with repair CNOTs"] = dnn_with_ro_CNOT

display(df)
df.to_csv("models/test_dataset_result.csv",index=False)

