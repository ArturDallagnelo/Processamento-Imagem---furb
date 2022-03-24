from matplotlib import pyplot as plt
import numpy as np
import cv2

imagem = cv2.imread('imagem.png')
plt.imshow(imagem, 'gray')
plt.title('Primeira Imagem')
plt.show()

marcador = np.ones((5, 5), np.uint8)
abrir = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, marcador)
fechar = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, marcador)
aHat = cv2.subtract(imagem, abrir)
fHat = cv2.subtract(fechar, imagem)
abrirImagem = cv2.add(imagem, aHat)
imagemDestaque = cv2.subtract(abrirImagem, fHat)
dilatar = cv2.dilate(imagem, marcador, iterations=1)
microCalcificacao = cv2.subtract(imagemDestaque, dilatar)
(T, aplicacao) = cv2.threshold(microCalcificacao, 50, 255, cv2.THRESH_BINARY)
plt.imshow(aplicacao, 'gray')
plt.title('Ultima Imagem')
plt.show()
