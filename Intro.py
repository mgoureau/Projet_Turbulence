import matplotlib.pyplot as plt
import numpy as np

#1
with open('./champs/champ0001.dat', 'r') as f:
    champs1 = f.readlines()
champs1 = [x.strip() for x in champs1]
X1,Y1,u1,v1 = [],[],[],[]
for line in champs1[2::]:
    X1.append(float(line.split()[0]))
    Y1.append(float(line.split()[1]))
    u1.append(float(line.split()[2]))
    v1.append(float(line.split()[3]))
X1,Y1,u1,v1=np.array(X1),np.array(Y1),np.array(u1),np.array(v1)
#2
with open('./champs/champ0002.dat', 'r') as f:
    champs2 = f.readlines()

champs2 = [x.strip() for x in champs2]
X2,Y2,u2,v2 = [],[],[],[]
for line in champs2[2::]:
    X2.append(float(line.split()[0]))
    Y2.append(float(line.split()[1]))
    u2.append(float(line.split()[2]))
    v2.append(float(line.split()[3]))
X2,Y2,u2,v2=np.array(X2),np.array(Y2),np.array(u2),np.array(v2)
#3
with open('./champs/champ0003.dat', 'r') as f:    
    champs3 = f.readlines()
X3,Y3,u3,v3 = [],[],[],[]
for line in champs3[2::]:
    X3.append(float(line.split()[0]))
    Y3.append(float(line.split()[1]))
    u3.append(float(line.split()[2]))
    v3.append(float(line.split()[3]))
X3,Y3,u3,v3=np.array(X3),np.array(Y3),np.array(u3),np.array(v3)
#4
with open('./champs/champ0004.dat', 'r') as f:
    champs4 = f.readlines()
X4,Y4,u4,v4 = [],[],[],[]
for line in champs4[2::]:
    X4.append(float(line.split()[0]))
    Y4.append(float(line.split()[1]))
    u4.append(float(line.split()[2]))
    v4.append(float(line.split()[3]))
X4,Y4,u4,v4=np.array(X4),np.array(Y4),np.array(u4),np.array(v4)
#5
with open('./champs/champ0005.dat', 'r') as f:
    champs5 = f.readlines()
X5,Y5,u5,v5 = [],[],[],[]
for line in champs5[2::]:
    X5.append(float(line.split()[0]))
    Y5.append(float(line.split()[1]))
    u5.append(float(line.split()[2]))
    v5.append(float(line.split()[3]))
X5,Y5,u5,v5=np.array(X5),np.array(Y5),np.array(u5),np.array(v5)
#6
with open('./champs/champ0006.dat', 'r') as f:
    champs6 = f.readlines()
X6,Y6,u6,v6 = [],[],[],[]
for line in champs6[2::]:
    X6.append(float(line.split()[0]))
    Y6.append(float(line.split()[1]))
    u6.append(float(line.split()[2]))
    v6.append(float(line.split()[3]))
X6,Y6,u6,v6=np.array(X6),np.array(Y6),np.array(u6),np.array(v6)
#7
with open('./champs/champ0007.dat', 'r') as f:
    champs7 = f.readlines()
X7,Y7,u7,v7 = [],[],[],[]
for line in champs7[2::]:
    X7.append(float(line.split()[0]))
    Y7.append(float(line.split()[1]))
    u7.append(float(line.split()[2]))
    v7.append(float(line.split()[3]))
X7,Y7,u7,v7=np.array(X7),np.array(Y7),np.array(u7),np.array(v7)
#8
with open('./champs/champ0008.dat', 'r') as f:
    champs8 = f.readlines()
X8,Y8,u8,v8 = [],[],[],[]
for line in champs8[2::]:
    X8.append(float(line.split()[0]))
    Y8.append(float(line.split()[1]))
    u8.append(float(line.split()[2]))
    v8.append(float(line.split()[3]))
X8,Y8,u8,v8=np.array(X8),np.array(Y8),np.array(u8),np.array(v8)
#9
with open('./champs/champ0009.dat', 'r') as f:
    champs9 = f.readlines()
X9,Y9,u9,v9 = [],[],[],[]
for line in champs9[2::]:
    X9.append(float(line.split()[0]))
    Y9.append(float(line.split()[1]))
    u9.append(float(line.split()[2]))
    v9.append(float(line.split()[3]))
X9,Y9,u9,v9=np.array(X9),np.array(Y9),np.array(u9),np.array(v9)
#10
with open('./champs/champ0010.dat', 'r') as f:
    champs10 = f.readlines()
X10,Y10,u10,v10 = [],[],[],[]
for line in champs10[2::]:
    X10.append(float(line.split()[0]))
    Y10.append(float(line.split()[1]))
    u10.append(float(line.split()[2]))
    v10.append(float(line.split()[3]))
X10,Y10,u10,v10=np.array(X10),np.array(Y10),np.array(u10),np.array(v10)
#11
with open('./champs/champ0011.dat', 'r') as f:
    champs11 = f.readlines()
X11,Y11,u11,v11 = [],[],[],[]
for line in champs11[2::]:
    X11.append(float(line.split()[0]))
    Y11.append(float(line.split()[1]))
    u11.append(float(line.split()[2]))
    v11.append(float(line.split()[3]))
X11,Y11,u11,v11=np.array(X11),np.array(Y11),np.array(u11),np.array(v11)
#12
with open('./champs/champ0012.dat', 'r') as f:
    champs12 = f.readlines()
X12,Y12,u12,v12 = [],[],[],[]
for line in champs12[2::]:
    X12.append(float(line.split()[0]))
    Y12.append(float(line.split()[1]))
    u12.append(float(line.split()[2]))
    v12.append(float(line.split()[3]))
X12,Y12,u12,v12=np.array(X12),np.array(Y12),np.array(u12),np.array(v12)
#13
with open('./champs/champ0013.dat', 'r') as f:
    champs13 = f.readlines()
X13,Y13,u13,v13 = [],[],[],[]
for line in champs13[2::]:
    X13.append(float(line.split()[0]))
    Y13.append(float(line.split()[1]))
    u13.append(float(line.split()[2]))
    v13.append(float(line.split()[3]))
X13,Y13,u13,v13=np.array(X13),np.array(Y13),np.array(u13),np.array(v13)
#14
with open('./champs/champ0014.dat', 'r') as f:
    champs14 = f.readlines()
X14,Y14,u14,v14 = [],[],[],[]
for line in champs14[2::]:
    X14.append(float(line.split()[0]))
    Y14.append(float(line.split()[1]))
    u14.append(float(line.split()[2]))
    v14.append(float(line.split()[3]))
X14,Y14,u14,v14=np.array(X14),np.array(Y14),np.array(u14),np.array(v14)
#15
with open('./champs/champ0015.dat', 'r') as f:
    champs15 = f.readlines()
X15,Y15,u15,v15 = [],[],[],[]
for line in champs15[2::]:

    X15.append(float(line.split()[0]))
    Y15.append(float(line.split()[1]))
    u15.append(float(line.split()[2]))
    v15.append(float(line.split()[3]))
X15,Y15,u15,v15=np.array(X15),np.array(Y15),np.array(u15),np.array(v15)
#16
with open('./champs/champ0016.dat', 'r') as f:
    champs16 = f.readlines()
X16,Y16,u16,v16 = [],[],[],[]
for line in champs16[2::]:
    X16.append(float(line.split()[0]))
    Y16.append(float(line.split()[1]))
    u16.append(float(line.split()[2]))
    v16.append(float(line.split()[3]))
X16,Y16,u16,v16=np.array(X16),np.array(Y16),np.array(u16),np.array(v16)
#17
with open('./champs/champ0017.dat', 'r') as f:
    champs17 = f.readlines()
X17,Y17,u17,v17 = [],[],[],[]
for line in champs17[2::]:
    X17.append(float(line.split()[0]))
    Y17.append(float(line.split()[1]))
    u17.append(float(line.split()[2]))
    v17.append(float(line.split()[3]))
X17,Y17,u17,v17=np.array(X17),np.array(Y17),np.array(u17),np.array(v17)
#18
with open('./champs/champ0018.dat', 'r') as f:
    champs18 = f.readlines()
X18,Y18,u18,v18 = [],[],[],[]
for line in champs18[2::]:
    X18.append(float(line.split()[0]))
    Y18.append(float(line.split()[1]))
    u18.append(float(line.split()[2]))
    v18.append(float(line.split()[3]))
X18,Y18,u18,v18=np.array(X18),np.array(Y18),np.array(u18),np.array(v18)
#19
with open('./champs/champ0019.dat', 'r') as f:
    champs19 = f.readlines()
X19,Y19,u19,v19 = [],[],[],[]
for line in champs19[2::]:
    X19.append(float(line.split()[0]))
    Y19.append(float(line.split()[1]))
    u19.append(float(line.split()[2]))
    v19.append(float(line.split()[3]))
X19,Y19,u19,v19=np.array(X19),np.array(Y19),np.array(u19),np.array(v19)
#20
with open('./champs/champ0020.dat', 'r') as f:
    champs20 = f.readlines()
X20,Y20,u20,v20 = [],[],[],[]
for line in champs20[2::]:
    X20.append(float(line.split()[0]))
    Y20.append(float(line.split()[1]))
    u20.append(float(line.split()[2]))
    v20.append(float(line.split()[3]))
X20,Y20,u20,v20=np.array(X20),np.array(Y20),np.array(u20),np.array(v20)

#Isocontours
plt.figure(1)
x1, y1 = np.meshgrid(np.unique(X1), np.unique(Y1))
plt.contour(x1, y1, u1.reshape(x1.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(2)
x2, y2 = np.meshgrid(np.unique(X2), np.unique(Y2))
plt.contour(x2, y2, u2.reshape(x2.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(3)
x3, y3 = np.meshgrid(np.unique(X3), np.unique(Y3))
plt.contour(x3, y3, u3.reshape(x3.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(4)
x4, y4 = np.meshgrid(np.unique(X4), np.unique(Y4))
plt.contour(x4, y4, u4.reshape(x4.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(5)
x5, y5 = np.meshgrid(np.unique(X5), np.unique(Y5))
plt.contour(x5, y5, u5.reshape(x5.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(6)
x6, y6 = np.meshgrid(np.unique(X6), np.unique(Y6))
plt.contour(x6, y6, u6.reshape(x6.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(7)
x7, y7 = np.meshgrid(np.unique(X7), np.unique(Y7))
plt.contour(x7, y7, u7.reshape(x7.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(8)
x8, y8 = np.meshgrid(np.unique(X8), np.unique(Y8))
plt.contour(x8, y8, u8.reshape(x8.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(9)
x9, y9 = np.meshgrid(np.unique(X9), np.unique(Y9))
plt.contour(x9, y9, u9.reshape(x9.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(10)
x10, y10 = np.meshgrid(np.unique(X10), np.unique(Y10))
plt.contour(x10, y10, u10.reshape(x10.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(11)
x11, y11 = np.meshgrid(np.unique(X11), np.unique(Y11))
plt.contour(x11, y11, u11.reshape(x11.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(12)
x12, y12 = np.meshgrid(np.unique(X12), np.unique(Y12))
plt.contour(x12, y12, u12.reshape(x12.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(13)
x13, y13 = np.meshgrid(np.unique(X13), np.unique(Y13))
plt.contour(x13, y13, u13.reshape(x13.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(14)
x14, y14 = np.meshgrid(np.unique(X14), np.unique(Y14))
plt.contour(x14, y14, u14.reshape(x14.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(15)
x15, y15 = np.meshgrid(np.unique(X15), np.unique(Y15))
plt.contour(x15, y15, u15.reshape(x15.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(16)
x16, y16 = np.meshgrid(np.unique(X16), np.unique(Y16))
plt.contour(x16, y16, u16.reshape(x16.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(17)
x17, y17 = np.meshgrid(np.unique(X17), np.unique(Y17))
plt.contour(x17, y17, u17.reshape(x17.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(18)
x18, y18 = np.meshgrid(np.unique(X18), np.unique(Y18))
plt.contour(x18, y18, u18.reshape(x18.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(19)
x19, y19 = np.meshgrid(np.unique(X19), np.unique(Y19))
plt.contour(x19, y19, u19.reshape(x19.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(20)
x20, y20 = np.meshgrid(np.unique(X20), np.unique(Y20))
plt.contour(x20, y20, u20.reshape(x20.shape),10, cmap='coolwarm')
plt.xlabel('x')
plt.ylabel('y')

# Afficher le graphique
plt.show()

