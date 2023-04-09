import numpy as np
import matplotlib.pyplot as plt
import os

VecU = np.zeros((71,1632))
VecV = np.zeros((71,1632))
VecX = np.zeros((71))
VecUmoy = np.zeros((71))

### Lecture des données
folder_path = './signaux/'
file_names = sorted(os.listdir(folder_path))

def recupereDonnees(chemin):
    file_path = os.path.join(folder_path, chemin)
    with open(file_path, 'r') as f:
        signal1 = f.readlines()
    signal1 = [x.strip() for x in signal1]
    u,v = [],[]
    x = float(signal1[1].split()[0])
    for line in signal1[5::]:
        u.append(float(line.split()[0]))
        v.append(float(line.split()[1]))
    u,v=np.array(u),np.array(v)
    return u,v,x

for i in range(71):
    u,v,x = recupereDonnees(file_names[i])
    VecU[i] = u - np.mean(u)
    VecV[i] = v
    VecX[i] = x
    VecUmoy[i] = (np.mean(((u-np.mean(u))**2)))**0.5

### Point de référence i=37, j = 26
u_0,v_0,x_0 = VecU[36],VecV[36],VecX[36]

r = VecX-x_0

dt = 0.025
N = 1632
K = 50
liste_k = np.arange(-K,K,1)
liste_kpos = np.arange(0,K,1)
tau_k = liste_k * dt

VecR = np.zeros((len(tau_k),len(r)))

def MoyTempDisc(N,k,u_0,u_i):
    somme = 0
    for n in range(N-k):
        somme += u_0[n]*u_i[(n+k)]
    return somme/(N-k)

for n in range(71):
    u_i = VecU[n]
    for k in liste_kpos:
        MoyTempDisc_i = MoyTempDisc(N,k,u_0,u_i)
        VecR[K+k,n] = MoyTempDisc_i/(VecUmoy[n]*VecUmoy[36])
        VecR[K-k,n] = MoyTempDisc_i/(VecUmoy[n]*VecUmoy[36])

print(VecR)

meshtau, meshr = np.meshgrid(tau_k, r)
CS = plt.contourf(r, tau_k, VecR, 25 , cmap='coolwarm')
plt.xlabel('r')
plt.ylabel('tau')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('R')
plt.show()

#Tourbillons

folder_path = './champs/'
file_names = sorted(os.listdir(folder_path))

def recupereDonnees(chemin):
    with open(chemin, 'r') as f:
        champs1 = f.readlines()
    champs1 = [x.strip() for x in champs1]
    X,Y,u,v = [],[],[],[]
    for line in champs1[2::]:
        X.append(float(line.split()[0]))
        Y.append(float(line.split()[1]))
        u.append(float(line.split()[2]))
        v.append(float(line.split()[3]))
    X,Y,u,v=np.array(X),np.array(Y),np.array(u)-0.25,np.array(v)
    return X,Y,u,v

for file in file_names :
    file_path = os.path.join(folder_path, file)
    X1,Y1,u1,v1 = recupereDonnees(file_path)
    plt.figure(file)
    X, Y = np.meshgrid(np.unique(X1), np.unique(Y1))
    plt.contourf(X, Y, u1.reshape(X.shape),25)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()