import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

folder_path = './champs/'
file_names = sorted(os.listdir(folder_path))

tmaillage = 2

dt = 2.5

def recupereDonnees(chemin):
    file_path = os.path.join(folder_path, chemin)
    with open(file_path, 'r') as f:
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

def CalculChamp(X,Y,U,V,dt):
    Xnew,Ynew = CalculPosition(X,Y,U,V,dt)
    Unew = interpolation(U,X[0],Y[:,0],Xnew,Ynew)
    Vnew = interpolation(V,X[0],Y[:,0],Xnew,Ynew)
    return Xnew,Ynew,Unew,Vnew

def CalculPosition(X,Y,U,V,dt):
    Xnew = np.zeros(X.shape)
    Ynew = np.zeros(Y.shape)
    l,c = X.shape
    Xnew = X - U*dt
    Ynew = Y - V*dt
    for i in range(l):
        for j in range(c):
            if Xnew[i,j] < 0 or Xnew[i,j] > 67.2:
                Xnew[i,j] = X[i,j]
            if Ynew[i,j] < -9.5 or Ynew[i,j] > 9.9:
                Ynew[i,j] = Y[i,j]
    return Xnew,Ynew

def interpolation(data,X,Y,Xnext,Ynext):
    l,c = data.shape
    Vitnew = np.zeros_like(data)
    f = interpolate.RegularGridInterpolator((np.unique(Y),np.unique(X)), data, bounds_error=False, fill_value=None)
    for i in range(l):
        for j in range(c):
            Vitnew[i,j] = f((Ynext[i,j],Xnext[i,j]))
    return Vitnew

def augMatrice(Champ, Xbase,Ybase, Xaug, Yaug):
    interp_func = interpolate.interp2d(Xbase,Ybase, Champ, kind='linear')
    return interp_func(Xaug,Yaug)

x,y,u20,v20=recupereDonnees(file_names[-1])
x20=np.linspace(np.unique(x)[0],np.unique(x)[-1],len(np.unique(x))*tmaillage)
y20=np.linspace(np.unique(y)[0],np.unique(y)[-1],len(np.unique(y))*tmaillage)
xx,yy=np.meshgrid(np.unique(x),np.unique(y))
X20, Y20 = np.meshgrid(np.unique(x20), np.unique(y20))
U = u20.reshape(xx.shape)
V = v20.reshape(yy.shape)
U20 = augMatrice(U,xx[0],yy[:,0],X20[0],Y20[:,0])
V20 = augMatrice(V,xx[0],yy[:,0],X20[0],Y20[:,0])

# #Affichage du champ de vitesse
plt.contourf(X20, Y20, U20,20, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

X19,Y19=CalculPosition(X20,Y20,U20,V20,dt)
u19,v19 = recupereDonnees(file_names[1])[2].reshape(xx.shape),recupereDonnees(file_names[1])[3].reshape(xx.shape)
U19,V19 = augMatrice(u19,xx[0],yy[:,0],X19[0],Y19[:,0]).reshape(X19.shape),augMatrice(v19,xx[0],yy[:,0],X19[0],Y19[:,0]).reshape(Y19.shape)
U19interp,V19interp = interpolation(U19,X20[0],Y20[:,0],X19,Y19),interpolation(V19,X20[0],Y20[:,0],X19,Y19)
print("19")
X18,Y18=CalculPosition(X19,Y19,U19interp,V19interp,dt)
u18,v18 = recupereDonnees(file_names[2])[2].reshape(xx.shape),recupereDonnees(file_names[2])[3].reshape(xx.shape)
U18,V18 = augMatrice(u18,xx[0],yy[:,0],X18[0],Y18[:,0]).reshape(X18.shape),augMatrice(v18,xx[0],yy[:,0],X18[0],Y18[:,0]).reshape(Y18.shape)
U18interp,V18interp = interpolation(U18,X19[0],Y19[:,0],X18,Y18),interpolation(V18,X19[0],Y19[:,0],X18,Y18)
print("18")
X17,Y17=CalculPosition(X18,Y18,U18interp,V18interp,dt)
u17,v17 = recupereDonnees(file_names[3])[2].reshape(xx.shape),recupereDonnees(file_names[3])[3].reshape(xx.shape)
U17,V17 = augMatrice(u17,xx[0],yy[:,0],X17[0],Y17[:,0]).reshape(X17.shape),augMatrice(v17,xx[0],yy[:,0],X17[0],Y17[:,0]).reshape(Y17.shape)
U17interp,V17interp = interpolation(U17,X18[0],Y18[:,0],X17,Y17),interpolation(V17,X18[0],Y18[:,0],X17,Y17)
print("17")
X16,Y16=CalculPosition(X17,Y17,U17interp,V17interp,dt)
u16,v16 = recupereDonnees(file_names[4])[2].reshape(xx.shape),recupereDonnees(file_names[4])[3].reshape(xx.shape)
U16,V16 = augMatrice(u16,xx[0],yy[:,0],X16[0],Y16[:,0]).reshape(X16.shape),augMatrice(v16,xx[0],yy[:,0],X16[0],Y16[:,0]).reshape(Y16.shape)
U16interp,V16interp = interpolation(U16,X17[0],Y17[:,0],X16,Y16),interpolation(V16,X17[0],Y17[:,0],X16,Y16)
print("16")
X15,Y15=CalculPosition(X16,Y16,U16interp,V16interp,dt)
u15,v15 = recupereDonnees(file_names[5])[2].reshape(xx.shape),recupereDonnees(file_names[5])[3].reshape(xx.shape)
U15,V15 = augMatrice(u15,xx[0],yy[:,0],X15[0],Y15[:,0]).reshape(X15.shape),augMatrice(v15,xx[0],yy[:,0],X15[0],Y15[:,0]).reshape(Y15.shape)
U15interp,V15interp = interpolation(U15,X16[0],Y16[:,0],X15,Y15),interpolation(V15,X16[0],Y16[:,0],X15,Y15)
print("15")
X14,Y14=CalculPosition(X15,Y15,U15interp,V15interp,dt)
u14,v14 = recupereDonnees(file_names[6])[2].reshape(xx.shape),recupereDonnees(file_names[6])[3].reshape(xx.shape)
U14,V14 = augMatrice(u14,xx[0],yy[:,0],X14[0],Y14[:,0]).reshape(X14.shape),augMatrice(v14,xx[0],yy[:,0],X14[0],Y14[:,0]).reshape(Y14.shape)
U14interp,V14interp = interpolation(U14,X15[0],Y15[:,0],X14,Y14),interpolation(V14,X15[0],Y15[:,0],X14,Y14)
print("14")
X13,Y13=CalculPosition(X14,Y14,U14interp,V14interp,dt)
u13,v13 = recupereDonnees(file_names[7])[2].reshape(xx.shape),recupereDonnees(file_names[7])[3].reshape(xx.shape)
U13,V13 = augMatrice(u13,xx[0],yy[:,0],X13[0],Y13[:,0]).reshape(X13.shape),augMatrice(v13,xx[0],yy[:,0],X13[0],Y13[:,0]).reshape(Y13.shape)
U13interp,V13interp = interpolation(U13,X14[0],Y14[:,0],X13,Y13),interpolation(V13,X14[0],Y14[:,0],X13,Y13)
print("13")
X12,Y12=CalculPosition(X13,Y13,U13interp,V13interp,dt)
u12,v12 = recupereDonnees(file_names[8])[2].reshape(xx.shape),recupereDonnees(file_names[8])[3].reshape(xx.shape)
U12,V12 = augMatrice(u12,xx[0],yy[:,0],X12[0],Y12[:,0]).reshape(X12.shape),augMatrice(v12,xx[0],yy[:,0],X12[0],Y12[:,0]).reshape(Y12.shape)
U12interp,V12interp = interpolation(U12,X13[0],Y13[:,0],X12,Y12),interpolation(V12,X13[0],Y13[:,0],X12,Y12)
print("12")
X11,Y11=CalculPosition(X12,Y12,U12interp,V12interp,dt)
u11,v11 = recupereDonnees(file_names[9])[2].reshape(xx.shape),recupereDonnees(file_names[9])[3].reshape(xx.shape)
U11,V11 = augMatrice(u11,xx[0],yy[:,0],X11[0],Y11[:,0]).reshape(X11.shape),augMatrice(v11,xx[0],yy[:,0],X11[0],Y11[:,0]).reshape(Y11.shape)
U11interp,V11interp = interpolation(U11,X12[0],Y12[:,0],X11,Y11),interpolation(V11,X12[0],Y12[:,0],X11,Y11)
print("11")
X10,Y10=CalculPosition(X11,Y11,U11interp,V11interp,dt)
u10,v10 = recupereDonnees(file_names[10])[2].reshape(xx.shape),recupereDonnees(file_names[10])[3].reshape(xx.shape)
U10,V10 = augMatrice(u10,xx[0],yy[:,0],X10[0],Y10[:,0]).reshape(X10.shape),augMatrice(v10,xx[0],yy[:,0],X10[0],Y10[:,0]).reshape(Y10.shape)
U10interp,V10interp = interpolation(U10,X11[0],Y11[:,0],X10,Y10),interpolation(V10,X11[0],Y11[:,0],X10,Y10)
print("10")
X9,Y9=CalculPosition(X10,Y10,U10interp,V10interp,dt)
u9,v9 = recupereDonnees(file_names[11])[2].reshape(xx.shape),recupereDonnees(file_names[11])[3].reshape(xx.shape)
U9,V9 = augMatrice(u9,xx[0],yy[:,0],X9[0],Y9[:,0]).reshape(X9.shape),augMatrice(v9,xx[0],yy[:,0],X9[0],Y9[:,0]).reshape(Y9.shape)
U9interp,V9interp = interpolation(U9,X10[0],Y10[:,0],X9,Y9),interpolation(V9,X10[0],Y10[:,0],X9,Y9)

X8,Y8=CalculPosition(X9,Y9,U9interp,V9interp,dt)
u8,v8 = recupereDonnees(file_names[12])[2].reshape(xx.shape),recupereDonnees(file_names[12])[3].reshape(xx.shape)
U8,V8 = augMatrice(u8,xx[0],yy[:,0],X8[0],Y8[:,0]).reshape(X8.shape),augMatrice(v8,xx[0],yy[:,0],X8[0],Y8[:,0]).reshape(Y8.shape)
U8interp,V8interp = interpolation(U8,X9[0],Y9[:,0],X8,Y8),interpolation(V8,X9[0],Y9[:,0],X8,Y8)

X7,Y7=CalculPosition(X8,Y8,U8interp,V8interp,dt)
u7,v7 = recupereDonnees(file_names[13])[2].reshape(xx.shape),recupereDonnees(file_names[13])[3].reshape(xx.shape)
U7,V7 = augMatrice(u7,xx[0],yy[:,0],X7[0],Y7[:,0]).reshape(X7.shape),augMatrice(v7,xx[0],yy[:,0],X7[0],Y7[:,0]).reshape(Y7.shape)
U7interp,V7interp = interpolation(U7,X8[0],Y8[:,0],X7,Y7),interpolation(V7,X8[0],Y8[:,0],X7,Y7)

X6,Y6=CalculPosition(X7,Y7,U7interp,V7interp,dt)
u6,v6 = recupereDonnees(file_names[14])[2].reshape(xx.shape),recupereDonnees(file_names[14])[3].reshape(xx.shape)
U6,V6 = augMatrice(u6,xx[0],yy[:,0],X6[0],Y6[:,0]).reshape(X6.shape),augMatrice(v6,xx[0],yy[:,0],X6[0],Y6[:,0]).reshape(Y6.shape)
U6interp,V6interp = interpolation(U6,X7[0],Y7[:,0],X6,Y6),interpolation(V6,X7[0],Y7[:,0],X6,Y6)

X5,Y5=CalculPosition(X6,Y6,U6interp,V6interp,dt)
u5,v5 = recupereDonnees(file_names[15])[2].reshape(xx.shape),recupereDonnees(file_names[15])[3].reshape(xx.shape)
U5,V5 = augMatrice(u5,xx[0],yy[:,0],X5[0],Y5[:,0]).reshape(X5.shape),augMatrice(v5,xx[0],yy[:,0],X5[0],Y5[:,0]).reshape(Y5.shape)
U5interp,V5interp = interpolation(U5,X6[0],Y6[:,0],X5,Y5),interpolation(V5,X6[0],Y6[:,0],X5,Y5)

X4,Y4=CalculPosition(X5,Y5,U5interp,V5interp,dt)
u4,v4 = recupereDonnees(file_names[16])[2].reshape(xx.shape),recupereDonnees(file_names[16])[3].reshape(xx.shape)
U4,V4 = augMatrice(u4,xx[0],yy[:,0],X4[0],Y4[:,0]).reshape(X4.shape),augMatrice(v4,xx[0],yy[:,0],X4[0],Y4[:,0]).reshape(Y4.shape)
U4interp,V4interp = interpolation(U4,X5[0],Y5[:,0],X4,Y4),interpolation(V4,X5[0],Y5[:,0],X4,Y4)
print("4")
X3,Y3=CalculPosition(X4,Y4,U4interp,V4interp,dt)
u3,v3 = recupereDonnees(file_names[17])[2].reshape(xx.shape),recupereDonnees(file_names[17])[3].reshape(xx.shape)
U3,V3 = augMatrice(u3,xx[0],yy[:,0],X3[0],Y3[:,0]).reshape(X3.shape),augMatrice(v3,xx[0],yy[:,0],X3[0],Y3[:,0]).reshape(Y3.shape)
U3interp,V3interp = interpolation(U3,X4[0],Y4[:,0],X3,Y3),interpolation(V3,X4[0],Y4[:,0],X3,Y3)

X2,Y2=CalculPosition(X3,Y3,U3interp,V3interp,dt)
u2,v2 = recupereDonnees(file_names[18])[2].reshape(xx.shape),recupereDonnees(file_names[18])[3].reshape(xx.shape)
U2,V2 = augMatrice(u2,xx[0],yy[:,0],X2[0],Y2[:,0]).reshape(X2.shape),augMatrice(v2,xx[0],yy[:,0],X2[0],Y2[:,0]).reshape(Y2.shape)
U2interp,V2interp = interpolation(U2,X3[0],Y3[:,0],X2,Y2),interpolation(V2,X3[0],Y3[:,0],X2,Y2)

X1,Y1=CalculPosition(X2,Y2,U2interp,V2interp,dt)
u1,v1 = recupereDonnees(file_names[19])[2].reshape(xx.shape),recupereDonnees(file_names[19])[3].reshape(xx.shape)
U1,V1 = augMatrice(u1,xx[0],yy[:,0],X1[0],Y1[:,0]).reshape(X1.shape),augMatrice(v1,xx[0],yy[:,0],X1[0],Y1[:,0]).reshape(Y1.shape)
U1interp,V1interp = interpolation(U1,X2[0],Y2[:,0],X1,Y1),interpolation(V1,X2[0],Y2[:,0],X1,Y1)
print("1")



# #Affichage du champ de vitesse
plt.contourf(X1, Y1, U1,20, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

dx = (X20[0,1]-X20[0,0])
dy = (Y20[1,0]-Y20[0,0])

dX1dx20 = np.gradient(X1,dx,axis=1)
dX1dy20 = np.gradient(X1,dy,axis=0)

dY1dx20 = np.gradient(Y1,dx,axis=1)
dY1dy20 = np.gradient(Y1,dy,axis=0)

VPmax = np.zeros((dX1dx20.shape))

for i in range(dX1dx20.shape[0]):
    for j in range(dX1dx20.shape[1]):
        A = np.zeros((2,2))
        A[0,0] = dX1dx20[i,j]
        A[0,1] = dX1dy20[i,j]
        A[1,0] = dY1dx20[i,j]
        A[1,1] = dY1dy20[i,j]
        At= np.transpose(A)
        AtA = np.matmul(At,A)
        VPmax[i,j]=max(np.linalg.eigvalsh(AtA))

T = 19*dt

FTLE = np.log(VPmax)/(2*T)


CS=plt.contourf(X20, Y20, FTLE,25, cmap='coolwarm')
# plt.quiver(Xm20, Ym20, U20-0.5719103015997616, V20, color='k', scale=25)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('FTLE')
plt.xlabel('x')
plt.ylabel('y')
# plt.savefig('maillage4.eps', format='eps')
plt.show()

# Xm19 = Xm20 - U20*dt
# Ym19 = Ym20 - V20*dt

# # f=interpolate.RectBivariateSpline(Xm20,Ym20,U20,kx=2, ky=2)
# f = interpolate.RegularGridInterpolator((np.unique(Y20),np.unique(X20)), U20, bounds_error=False, fill_value=None)
# U19 = np.zeros_like(U20)


# def interpolation(Vitesse,X,Y,Xnext,Ynext):
#     l,c = Vitesse.shape
#     Vitnew = np.zeros((l,c))
#     f=interpolate.interp2d(X,Y,Vitesse,kind='cubic')
#     for i in range(l):
#         for j in range(c):
#             Vitnew[i,j] = f(Xnext[i,j],Ynext[i,j])
#     return Vitnew

# def CalculPosition(X,Y,U,V,dt):
#     Xnew = np.zeros(X.shape)
#     Ynew = np.zeros(Y.shape)
#     l,c = X.shape
#     for i in range(l):
#         for j in range(c):
#             Xnew[i,j] = X[i,j] - U[i,j]*dt
#             Ynew[i,j] = Y[i,j] - V[i,j]*dt
#             if Xnew[i,j] < 0 or Xnew[i,j] > 67.2/tmaillage:
#                 Xnew[i,j] = X[i,j]
#             if Ynew[i,j] < -9.5/tmaillage or Ynew[i,j] > 9.9/tmaillage:
#                 Ynew[i,j] = Y[i,j]
#     return Xnew,Ynew

# def CalculChamp(X,Y,U,V,dt):
#     Xnew,Ynew = CalculPosition(X,Y,U,V,dt)
#     Unew = interpolation(U,X,Y,Xnew,Ynew)
#     Vnew = interpolation(V,X,Y,Xnew,Ynew)
#     return Xnew,Ynew,Unew,Vnew

# import numpy as np
# from scipy.interpolate import interp2d

# # Créer une grille de départ avec des valeurs aléatoires
# grid_start = np.random.rand(39, 71)

# # Créer une fonction d'interpolation bilinéaire
# interp_func = interp2d(np.arange(0, 71), np.arange(0, 39), grid_start, kind='linear')

# # Créer la grille de destination avec la taille souhaitée
# new_grid = np.zeros((78, 142))

# # Évaluer la fonction d'interpolation bilinéaire sur chaque point de la grille de destination
# for i in range(142):
#     for j in range(78):
#         new_grid[j, i] = interp_func(j * 39/77, i * 71/141)

