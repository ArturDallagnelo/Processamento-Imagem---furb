from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)
data = load_iris()

X=data.data
y=data.target
X_outliers = rng.uniform(low=-4, high=4, size=(X.shape[0], X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

clf = IsolationForest()
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

print(y_pred_test)
print(y_pred_outliers)


# #Trabalho 02 Artur Dallagnelo e MAik Carminati
# from matplotlib import pyplot as plt
# import numpy as np
# import cv2
#
# imagem = cv2.imread('iris.jpg')
# plt.imshow(imagem, 'gray')
# plt.title('Primeira Imagem')
# plt.show()
# marcador = np.ones((5, 5), np.uint8)
# abrir = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, marcador)
# fechar = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, marcador)
# aHat = cv2.subtract(imagem, abrir)
# fHat = cv2.subtract(fechar, imagem)
# abrirImagem = cv2.add(imagem, aHat)
# imagemDestaque = cv2.subtract(abrirImagem, fHat)
# dilatar = cv2.dilate(imagem, marcador, iterations=1)
# # microCalcificacao = cv2.subtract(imagemDestaque, dilatar)
# # (T, aplicacao) = cv2.threshold(microCalcificacao, 50, 255, cv2.THRESH_BINARY)
# # plt.imshow(aplicacao, 'gray')
# # plt.title('Ultima Imagem')
# # plt.show()
#
