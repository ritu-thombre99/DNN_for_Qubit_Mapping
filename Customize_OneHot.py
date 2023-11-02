import numpy as np
def customize_OH(vec):
    '''Funzione che ritorna un customize one-hot codificando i Nan come 0'''
    length = len(vec)
    label = []
    for n in vec:
        l = []
        if n == None:
            for zero in range(length):
                l.append(0)
        else:
            l = num_to_vec(n, length)
        label = label + l

    return label

def customize_OH_withNan(vec, num_qubits):
    '''Funzione che ritorna un customize one-hot codificando i Nan come 5'''
    vec = list(vec)
    length = len(vec)+1
    label = []
    for n in vec:
        l = []
        if n in range(0,num_qubits):
            l = num_to_vec(n, length)
        else:
            #print('ciao')
            l = num_to_vec(n=num_qubits, len_vec=length)

        label = label + l
    return label

def num_to_vec(n,len_vec):
    '''Funzione che ritorna un vettore di lunghezza len_vec di tutti 0 e un 1 in posizione n'''
    l=[]
    for i in range(len_vec):
        if i != n:
            l.append(0)
        else:
            l.append(1)
    return l


def check_in_list(i, list):
    '''funzione che mi dice in che posizione si trovano gli elementi i nella lista list'''
    index = []
    for j in range(len(list)):
        if i == list[j]:
            index.append(j)
        else:
            continue

    return index


def give_mapping(vec,N_phys_qubits=5):
    layout=[]
    i=0
    for n in range(0, len(vec), N_phys_qubits):
        #print(n)
        if max(vec[n:n+N_phys_qubits])!=0:
            layout.append(vec.index(max(vec[n:n+N_phys_qubits]))-i*N_phys_qubits)
        else:
            layout.append(None)
        i=i+1

    return layout

#def takeSecond(elem):
 #   return elem[1][0]


def check_layout(vec, N_qubits):
    '''Funziona che controlla che il numero di qubit virtuali mappati, sia effettivamente
    uguale a quello che il circuito da mappare usa, e che non ci siano ripetizioni:
    nel caso ci siano torna una lista con l'indice del qubit virtuale e dove questo è stato ripetuto
    '''
    if len(vec)-vec.count(np.nan) == N_qubits:
        for virtual_qubit in vec:
            if vec.count(virtual_qubit) ==1 and virtual_qubit!= np.nan:
                continue
            else:
                return [virtual_qubit, check_in_list(vec, virtual_qubit)]
        return []
    else:
        return ['error']



def prob_mapping(vec, N_qubits,  N_slot=5):
    '''Funziona che torna il layout in base alla probabilità del softmax:
    fissato il numero di qubit mi ritorna un vettore le cui componenti coincidono con le slot fisiche, e
    il cui valore corrisponde al qubit virtuale da mappare.
    NON C E' NESSUN CONTROLLO SUL MAPPING: SO SOLO CHE SE HO 3 QUBIT VIRTUALI DEVO MAPPARNE ALTRETTANTI,
    Ma potrebbero essere anche tutti 0 o 1'''
    layout = []
    max_prob = []
    slots = []
    for i in range(N_slot):
        layout.append(np.nan)
    for n in range(0,N_slot):
        l = vec[n*N_slot:N_slot+n*N_slot]
        slots.append(l)
        max_prob.append([n,max(l)])

    max_prob.sort(key=lambda i:i[1],reverse=True)
    for j in range(N_qubits):
        slot = max_prob[j][0]
        virtual_qubit = slots[slot].index(max_prob[j][1])
        layout[slot] = virtual_qubit

    return layout


def prob_mapping_2(vec, N_qubits, N_slot=5):
    layout = []
    slots = []
    for i in range(N_slot):
        layout.append(np.nan)

    for n in range(0, N_slot):
        l = vec[n * N_slot:N_slot + n * N_slot]
        slots.append(l)#In slots ci sono i 5 sottovettori contenenti le rispettive probabilità

    #Ciclo per vedere quale qubit virtuale ha la probabilità più alta, cosi da cominciare il mapping da lui
    mapping_order=[]
    for v_q in range(N_qubits):
        ordine = []
        for probs in slots:
            ordine.append(probs[v_q])
        #print(ordine)
        mapping_order.append([v_q, max(ordine)])
    mapping_order.sort(key=lambda i:i[1], reverse=True)

    #print(mapping_order)
    for iteration in range(N_qubits):#faccio un ciclo su quanti sono i qubit virtuali da mappare
        v_qubit = mapping_order[iteration][0]
        #print('v_qubit', v_qubit)
        prob_v_q = []
        for prob in slots:
            prob_v_q.append(prob[v_qubit])#Metto in prob_v_q tutte le probabilità relative al qubit virtuale in iterazione
        #Ordino queste probabilità in maniera decrescente
        prob_v_q_ord = prob_v_q.copy()
        prob_v_q_ord.sort(reverse=True)
        #print(prob_v_q)
        #print(prob_v_q_ord)
        check = 0
        j = 0
        while check==0 and j<N_slot:
            #faccio un ciclo prendendomi la probabilità più alta e andando a mettere il qubit
            #virtuale nello slot relativo alla prob virtuale. Se Questo slot è già occupato
            #Passo alla seconda probabilità più alta e cosi via
            slot = prob_v_q.index(prob_v_q_ord[j])
            #print(slot, layout[slot])
            if layout[slot] is np.nan:
                #print('ciao')
                layout[slot] = v_qubit
                #print(layout)
                check = 1
            else:
                j=j+1

    return layout




def max_per_slot(vec, dim_slot=6, N_slot=5):
    for i in range(N_slot):
        max_ind = np.argmax(vec[i*dim_slot:i*dim_slot+dim_slot])
        vec = list(vec)
        for j in range(dim_slot):
            if max_ind == j:
                vec[j + i*dim_slot] = 1
            else:
                vec[j + i*dim_slot] = 0
    return np.array(vec)



''''
y = np.random.random(30)
print(y)
print(max_per_slot(y))
'''''
