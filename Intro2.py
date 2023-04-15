import numpy as np
import matplotlib.pyplot as plt
import os

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
    X,Y,u,v=np.array(X),np.array(Y),np.array(u),np.array(v)
    return X,Y,u,v

for file in file_names :
    file_path = os.path.join(folder_path, file)
    X1,Y1,u1,v1 = recupereDonnees(file_path)
    plt.figure(file)
    X, Y = np.meshgrid(np.unique(X1), np.unique(Y1))
    plt.contourf(X, Y, u1.reshape(X.shape),25,cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    