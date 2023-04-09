import numpy as np
import matplotlib.pyplot as plt

# Créer les listes x et y correspondant aux coordonnées de sommets du maillage
x = [0, 0, 1, 1, 2, 2]
y = [0, 1, 0, 1, 0, 1]

# Créer une grille de points X et Y correspondant aux coordonnées de sommets
X, Y = np.meshgrid(np.unique(x), np.unique(y))

# Créer une matrice u contenant les valeurs de la vitesse aux nœuds de chaque cellule
u = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7])

# Tracer les isocontours de la vitesse
plt.contour(X, Y, u.reshape(X.shape))

# Ajouter des étiquettes aux axes x et y
plt.xlabel('x')
plt.ylabel('y')

# Afficher le graphique
plt.show()
print(u.reshape(X.shape))
print(X)