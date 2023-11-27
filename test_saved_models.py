from get_test_datasets import return_dataset
test_dataset = return_dataset()

len(test_dataset)

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helper import *

mlp_edge_feature = tf.keras.models.load_model("models/mlp_edge_features.keras")
dnn = tf.keras.models.load_model("models/dnn.keras")
mlp = tf.keras.models.load_model("models/mlp.keras")


SS = StandardScaler()
num_qubits = 7
count_or = 0
count_nr = 0
count_mlp = 0
count_mlp_edge = 0
for i in tqdm(range(len(test_dataset))):
    qc = test_dataset[i]
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
        
print("DNN accuracy without repair operator:", count_or/len(test_dataset))
print("DNN accuracy with repair operator:", count_nr/len(test_dataset))
print("MLP accuracy:", count_mlp/len(test_dataset))
print("MLP with edge features accuracy:", count_mlp_edge/len(test_dataset))

