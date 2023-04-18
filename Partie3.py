import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

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
            if (X[i,j] < 0 and X[i,j] > 6.71391989E+01):
                Xnew[i,j] = X[i,j]
            if (Y[i,j] < -9.49012463 and Y[i,j] > 9.87634180):
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
    Vnew = interpolation(V,Ynew,Y)
    return Xnew,Ynew,Unew,Vnew

X19,Y19,U19,V19 = CalculChamp(X20,Y20,U20,V20,dt)
X18,Y18,U18,V18 = CalculChamp(X19,Y19,U19,V19,dt)
X17,Y17,U17,V17 = CalculChamp(X18,Y18,U18,V18,dt)
X16,Y16,U16,V16 = CalculChamp(X17,Y17,U17,V17,dt)
X15,Y15,U15,V15 = CalculChamp(X16,Y16,U16,V16,dt)
X14,Y14,U14,V14 = CalculChamp(X15,Y15,U15,V15,dt)
X13,Y13,U13,V13 = CalculChamp(X14,Y14,U14,V14,dt)
X12,Y12,U12,V12 = CalculChamp(X13,Y13,U13,V13,dt)
X11,Y11,U11,V11 = CalculChamp(X12,Y12,U12,V12,dt)
X10,Y10,U10,V10 = CalculChamp(X11,Y11,U11,V11,dt)
X9,Y9,U9,V9 = CalculChamp(X10,Y10,U10,V10,dt)
X8,Y8,U8,V8 = CalculChamp(X9,Y9,U9,V9,dt)
X7,Y7,U7,V7 = CalculChamp(X8,Y8,U8,V8,dt)
X6,Y6,U6,V6 = CalculChamp(X7,Y7,U7,V7,dt)
X5,Y5,U5,V5 = CalculChamp(X6,Y6,U6,V6,dt)
X4,Y4,U4,V4 = CalculChamp(X5,Y5,U5,V5,dt)
X3,Y3,U3,V3 = CalculChamp(X4,Y4,U4,V4,dt)
X2,Y2,U2,V2 = CalculChamp(X3,Y3,U3,V3,dt)
X1,Y1,U1,V1 = CalculChamp(X2,Y2,U2,V2,dt)

dx = (X20[0,1]-X20[0,0])
dy = (Y20[1,0]-Y20[0,0])

dX1dx20 = np.gradient(X1,dx,axis=1)
dX1dy20 = np.gradient(X1,dy,axis=0)

dY1dx20 = np.gradient(Y1,dx,axis=1)
dY1dy20 = np.gradient(Y1,dy,axis=0)

VPmax = np.zeros((dX1dy20.shape))

for i in range(dX1dx20.shape[0]):
    for j in range(dX1dx20.shape[1]):
            A = np.zeros((2,2))
            A[0,0] = dX1dx20[i,j]
            A[0,1] = dX1dy20[i,j]
            A[1,0] = dY1dx20[i,j]
            A[1,1] = dY1dy20[i,j]
            At= np.transpose(A)
            VPmax[i,j]=max(np.linalg.eigvals(np.dot(At,A)))

T = 19*dt

FTLE = np.log(VPmax)/(2*T)

CS=plt.contourf(X1, Y1, FTLE,25, cmap='coolwarm')
xmesh, ymesh = np.meshgrid(np.unique(X20), np.unique(Y20))
# plt.quiver(xmesh, ymesh, U20, V20, color='k', scale=75)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('FTLE')
plt.xlabel('x')
plt.ylabel('y')
plt.show()