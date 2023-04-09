import numpy as np
import matplotlib.pyplot as plt
import os

### Lecture des données
folder_path = './signaux/'
file_names = sorted(os.listdir(folder_path))

def recupereDonnees(chemin):
    file_path = os.path.join(folder_path, chemin)
    with open(file_path, 'r') as f:
        signal1 = f.readlines()
    signal1 = [x.strip() for x in signal1]
    u,v = [],[]
    for line in signal1[5::]:
        u.append(float(line.split()[0]))
        v.append(float(line.split()[1]))
    u,v=np.array(u),np.array(v)
    return u,v

### Point de référence i=37, j = 26

u_0,v_0 = recupereDonnees(file_names[36])
u1,v1 = recupereDonnees(file_names[0])


dt = 0.025

K = 200
liste_k = np.arange(-K,K,1)
liste_kpos = np.arange(0,K,1)
tau_k = liste_kpos * dt

def MoyTempDisc(N,k,u_0,u_i):
    somme = 0
    for n in range(N-k):
        somme += u_0[n]*u_i[(n+k)]
    return somme/(N-k)

R = np.zeros(len(liste_kpos))

for k in liste_kpos:
    moy = MoyTempDisc(K,k,u_0,u1)
    deno = np.sqrt(u_0[0]**2) * np.sqrt(u1[0]**2) 
    R[k] = moy/deno

plt.plot(tau_k,R)
plt.show()
