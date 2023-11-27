# Work Log
TODO:
1. Add a new evaluation metric: the number of CNOT gates (CNOT gates are required to
perform SWAPs) (2 hours): Create a different test dataset and test on it along with time taken to find the mapping
2. Plotting all the results and organizing them (5 hours)
3. Writing the final project report (10 hours)

DONE:
1. create the dataset using IBM’s publicly available 7-qubit quantum computers (ibm_perth, ibm_lagos, ibm_nairobi)
2. Reproduce the results1 on our 7-qubit dataset (10 hours)
3. add sabrelayout method in label
4. Created a new dataset for larger quantum circuits (7-qubit only but large depths).. This dataset includes circuits of different types,
such as Grover’s circuit, Shor’s circuit, Quantum Fourier Transform, and various types of GHZ state preparation 
5. save and test models (kinda done)

# DNN_for_Mapping
This is a repository in which you can find all the codes for implementing a Deep Neural Network to perform the mapping of virtual qubits 
of a quantum algorithm coded by a circuit into a physical processor's quantum bits. 

There are scripts for the dataset collection using Qiskit Ranom Circuit generator and scripts to build the neural network model using Keras. 
