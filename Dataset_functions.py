import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analisi_combinazioni(df_tot, diz=0):
    qubit_in_slot = []
    for i in range(5):
        #print(df_tot.groupby(str(i)).count())
        qubit_in_slot.append(df_tot.groupby(str(i)).count().sum()['Index'])

    from itertools import permutations
    comb = list(permutations([0, 1, 2, 3, 4]))
    comb = list(set(comb))
    #print(comb)
    ripetizioni = []
    df = df_tot.fillna(10)

    for lista in comb:
        df0 = df[df['0'] == lista[0]]
        df1 = df0[df0['1'] == lista[1]]
        df2 = df1[df1['2'] == lista[2]]
        df3 = df2[df2['3'] == lista[3]]
        df4 = df3[df3['4'] == lista[4]]
        ripetizioni.append(len(df4.sum(axis=1).tolist()))
    dict = {}

    for i in range(len(comb)):
        if ripetizioni[i]!=0:
            dict[comb[i]] = ripetizioni[i]

    #print(sum(dict.values()))

    keys = list(dict.keys())
    values = list(dict.values())
    kv = []
    for i in range(len(keys)):
        kv.append((keys[i],values[i]))
    kv.sort(key=lambda i:i[1])
    keys_ord = []
    values_ord = []
    for i in range(len(kv)):
        keys_ord.append(list(kv[i][0]))
        values_ord.append(kv[i][1])
    #print(keys_ord[-1], values_ord[-1])

    if diz == 0:
        return keys_ord
    else:
        return dict

def pulisci_dataset(df,N_rows_to_delete,layout):
    df_1 = df[df['0'] == layout[0]]
    df_1 = df_1[df_1['1'] == layout[1]]
    df_1 = df_1[df_1['2'] == layout[2]]
    df_1 = df_1[df_1['3'] == layout[3]]
    df_1 = df_1[df_1['4'] == layout[4]]
    ind = list(df_1.iloc[:, 0].values)
    for i in range(N_rows_to_delete):
        rand = np.random.randint(0, len(ind) - 1)
        df = df.drop(ind[rand])
        ind.pop(rand)
    return df


# def Histo_Dataset(df, N_qubits, color='blue', save=False, title=None):
#     '''
#     Function that returns the histograms with the occurrences of all the slots:
#     if N_qubits < 5 then the N_qubits index on the histogram indicates the Nan (since the
#     qubits to map range from (0 to N_qubits-1)
#     ________________________________________
#     '''
#     for i in range(5):
#         df1 = df.groupby(str(i)).count()
#         if N_qubits != 5:
#             df1 = df1.fillna(N_qubits)
#         y = list(df1.iloc[:,0].values)
#         x = [n for n in range(N_qubits)]
#         plt.bar(x,y,align='center', color=color)
#         plt.xlabel('virtual qubit')
#         plt.ylabel('Frequency_slot'+str(i))
#         plt.show()
#         if save == True:
#             plt.savefig(str(title)+'_'+str(i))


