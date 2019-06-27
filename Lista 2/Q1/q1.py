import cv2
import numpy as np

# leitura da imagem
img = cv2.imread('Thyroid.jpg', 0)

# cores escolhidas para o fatiamento (RGB)
colors = ((0, 0, 0), (0, 0, 205), (221, 160, 221), (255, 165, 0),
          (154, 205, 50), (240, 230, 140), (255, 0, 0), (255, 255, 255))

# tamanho da faixa de valores que será agrupada em cada cor
step = (np.max(img) + 1)//8 + 1

new_img = np.zeros((img.shape[0], img.shape[1], 3))
for l_idx, l in enumerate(img):
    for c_idx, c in enumerate(l):
        # atribuição da nova cor a cada pixel
        new_img[l_idx][c_idx] = np.flip(np.array(colors[c//step]))

# salva imagem do resultado
cv2.imwrite('intensity_sliced.png', new_img)

