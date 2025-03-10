# DNN for Qubit Mapping
We reproduce, extend and benchmark the DNN model for Qubit Mapping from [1]

Specifically, our contributions are:
* extended the DNN model from 5-qubits to 7-qubits by expanding the architecture naturally
* created a new dataset of 7-qubit random circuits, following the dataset format as described
* added Sabrelayout to the set of IBM algorithms for label generation
* ran the reimplemented DNN model on the newly produced dataset and reproduced results similar to what was expected
* created a new dataset of 7-qubit circuits of practical design and ran the DNN model on it without retraining to see if DNN can generalize
* benchmarked DNN against
    * the 2 best performing ML in [1]: RF-Gini and SVM-RBF
    * a shallow multiclass classifier
    * a geometrically motivated GNN model

## Code
### Models & Results
`dnn.py`                : Neural layout (Deep neural Network) model
`gnn-svm-rf.ipynb`      : Test accuracy results & model code for GNN model, SVM-RBF model and RF-Gini model.
`mlp_graph_features.py` : Shallow MLP classifier model
`models/mlp_edge_features_accuracy.txt`: Shallow MLP classifier training and test accuracy
`models/mlp_edge_features_accuracy.txt`: Neural Layout (Deep Neural Network) training and test accuracy
`models/figures.ipynd`  : Figures of various comparisons between models

### Datasets
`Dataset.py` : Generate dataset of random circuits (dataset is a csv file placed in `dataset/dataset_tesi/`)
`get_test_datasets.py` : Generate practical circuits (QFT,QPE,ALUs,GHZ,Grover) which are returned in a list when the function `return_dataset()` is called  

## References
[1] Giovanni Acampora and Roberto Schiattarella. Deep neural networks for quantum circuit mapping. Neural Computing and Applications, 33(20):13723–13743, 2021.

## Python and conda equirements
Conda environments:
conda create --name dnn_for_qubit_mapping -c anaconda python=3.8.18 
conda activate dnn_for_qubit_mapping
conda deactivate

To view environment in jupyter :
conda install nb_conda_kernels
conda install -n notebook_env nb_conda_kernels
