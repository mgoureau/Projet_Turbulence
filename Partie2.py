import numpy as np
import matplotlib.pyplot as plt
import os

with open('./champs/champ0020.dat', 'r') as f:
    champs20 = f.readlines()
X20,Y20,u20,v20 = [],[],[],[]
for line in champs20[2::]:
    X20.append(float(line.split()[0]))
    Y20.append(float(line.split()[1]))
    u20.append(float(line.split()[2]))
    v20.append(float(line.split()[3]))
X20,Y20,u20,v20=np.array(X20),np.array(Y20),np.array(u20),np.array(v20)

xmesh, ymesh = np.meshgrid(np.unique(X20), np.unique(Y20))

VecU = u20.reshape(xmesh.shape)
VecV = v20.reshape(xmesh.shape)

dUdx = np.diff(VecU, axis=0)/X20[1]
dUdy = np.diff(VecU, axis=1)/(Y20[71]-Y20[70])

dVdx = np.diff(VecV, axis=0)/X20[1]
dVdy = np.diff(VecV, axis=1)/(Y20[71]-Y20[70])

omega = dVdy[:-1] - dUdx[:,:-1]

Q = -0.5*(dUdx[:,:-1]**2 + dVdy[:-1]**2 + 2*dUdy[:-1]*dVdx[:,:-1])

#Affichage du champ de vitesse
plt.figure("Omega")
CS = plt.contourf(xmesh[:-1,:-1], ymesh[:-1,:-1], omega,25, cmap='coolwarm')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('w')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure("Q")
CS2 = plt.contourf(xmesh[:-1,:-1], ymesh[:-1,:-1], Q ,25, cmap='coolwarm')
cbar = plt.colorbar(CS2)
cbar.ax.set_ylabel('Q')
plt.xlabel('x')
plt.ylabel('y')
plt.show()