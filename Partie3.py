import numpy as np
import matplotlib.pyplot as plt

def interpolation(Vitesse,Positionnew,Positionold):
    l,c = Vitesse.shape
    Vitnew = np.zeros((l,c))
    for i in range(l):
        for j in range(c):
            x = Positionold[i]
            y = Vitesse[i]
            Vitnew[i,j] = np.interp(Positionnew[i,j],x,y)
    return Vitnew

def CalculPosition(X,Y,U,V,dt):
    Xnew = np.zeros(X.shape)
    Ynew = np.zeros(Y.shape)
    l,c = X.shape
    for i in range(l):
        for j in range(c):
            Xnew[i,j] = X[i,j] - U[i,j]*dt
            Ynew[i,j] = Y[i,j] - V[i,j]*dt
            if (X[i,j] < 0 and X[i,j] > 671):
                Xnew[i,j] = X[i,j]
            if (Y[i,j] < -9.49 and Y[i,j] > 9.876):
                Ynew[i,j] = Y[i,j]
    return Xnew,Ynew

with open('./champs/champ0020.dat', 'r') as f:
    champs20 = f.readlines()
x20,y20,u20,v20 = [],[],[],[]
for line in champs20[2::]:
    x20.append(float(line.split()[0]))
    y20.append(float(line.split()[1]))
    u20.append(float(line.split()[2]))
    v20.append(float(line.split()[3]))
X20,Y20,u20,v20=np.array(x20),np.array(y20),np.array(u20),np.array(v20)
X20, Y20 = np.meshgrid(np.unique(X20), np.unique(Y20))
U20 = u20.reshape(X20.shape)
V20 = v20.reshape(Y20.shape)
plt.contourf(X20, Y20, U20,25, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

dt = 25e-3

def CalculChamp(X,Y,U,V,dt):
    Xnew,Ynew = CalculPosition(X,Y,U,V,dt)
    Unew = interpolation(U,Xnew,X)
    print(Unew)
    Vnew = interpolation(V,Ynew,Y)
    return Xnew,Ynew,Unew,Vnew

X,Y,U,V = X20,Y20,U20,V20

for i in range(20):
    X,Y,U,V = CalculChamp(X,Y,U,V,dt)
    plt.contourf(X, Y, U,25, cmap='coolwarm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()