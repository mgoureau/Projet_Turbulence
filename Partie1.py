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

dt = 0.0025
N = 1632
K = 100
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
        MoyTempDisc_i2 = MoyTempDisc(N,k,u_i,u_0)
        VecR[K+k,n] = MoyTempDisc_i/(VecUmoy[n]*VecUmoy[36])
        VecR[K-k,n] = MoyTempDisc_i2/(VecUmoy[n]*VecUmoy[36])

meshtau, meshr = np.meshgrid(tau_k, r)
CS = plt.contourf(r, tau_k, VecR, 25 , cmap='coolwarm')
plt.xlabel('r')
plt.ylabel('tau')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('R')
plt.show()

VecUc = []
VecUc2 = []
for i in range(len(tau_k)):
    r_max = r[np.argmax(VecR[i])]*1e-3
    if max(VecR[i]) >= 0.3 and tau_k[i] != 0:
        VecUc2.append(r_max/tau_k[i])
Uc2 = np.mean(VecUc2)   

print("Uc2 =" , Uc2)

# for i in range(71):
#     signal = VecR[:,i]
#     r_max = r[i]*1e-3
#     if  max(signal) >= 0.3 and tau_k[np.argmax(signal)] != 0:
#         VecUc.append(abs(r_max)/abs(tau_k[np.argmax(signal)]))

for i in range(71):
     signal = VecR[:,i]
     if  max(signal) >= 0.1 and tau_k[np.argmax(signal)] != 0:
         VecUc.append(abs(r[i]*1e-3/tau_k[np.argmax(signal)]))
Uc1 = np.mean(VecUc)
print("Uc1 =" , Uc1)

#Uc=Uc1*2

Uc = Uc1

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
    X,Y,u,v=np.array(X),np.array(Y),np.array(u)-Uc,np.array(v)
    return X,Y,u,v

file = file_names[-1]
file_path = os.path.join(folder_path, file)
X1,Y1,u1,v1 = recupereDonnees(file_path)
plt.figure(file)
X, Y = np.meshgrid(np.unique(X1), np.unique(Y1))
plt.streamplot(X, Y, u1.reshape(X.shape), v1.reshape(X.shape),density=2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.quiver(X, Y, u1.reshape(X.shape), v1.reshape(X.shape),color='r',scale=20)
plt.xlabel('x')
plt.ylabel('y')
plt.show()