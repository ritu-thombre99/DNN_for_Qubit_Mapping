# Work Log
TODO:
1. Create a new dataset for larger quantum circuits (20-30 qubits) to be executed on IBM’s 127
qubit quantum computer (ibm_brisbane). This dataset will include circuits of different types,
such as Grover’s circuit, Shor’s circuit, Quantum Fourier Transform, and various types of
GHZ state preparation (5 hours)
2. Test Neural Layout for larger circuits (4 hours)
3. Add a new evaluation metric: the number of CNOT gates (CNOT gates are required to
perform SWAPs) (2 hours)
4. Plotting all the results and organizing them (5 hours)
5. Writing the final project report (10 hours)

DONE:
1. create the dataset using IBM’s publicly available 7-qubit quantum computers (ibm_perth, ibm_lagos, ibm_nairobi)
2. Reproduce the results1 on our 7-qubit dataset (10 hours)
3. add sabrelayout method in label

# DNN_for_Mapping
This is a repository in which you can find all the codes for implementing a Deep Neural Network to perform the mapping of virtual qubits 
of a quantum algorithm coded by a circuit into a physical processor's quantum bits. 

There are scripts for the dataset collection using Qiskit Ranom Circuit generator and scripts to build the neural network model using Keras. 
