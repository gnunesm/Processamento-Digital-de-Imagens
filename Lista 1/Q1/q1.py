import cv2
import numpy as np

img = cv2.imread('Fig8.02.jpg', 0)
thresh = 159

img[img > thresh] = 255
img[img <= thresh] = 0
cv2.imwrite('binary_sticks_.png', img)

img_marker = np.zeros(img.shape)
counts = []

current_stick = 1

def neighbor_marking(mark, line, column):
    marked = 0
    for l in range(line - 1, line + 2):
        for c in range(column - 1, column + 2):
            if img[l][c] != 0:
                if img_marker[m][n] == 0:
                    marked += neighbor_marking(mark, l, c)
    return marked

for m in range(img.shape[0]):
    for n in range(img.shape[1]):
        if img_marker[m][n] == 0:
            if img[m][n] != 0:
                counts.append(neighbor_marking(current_stick, m, n))
                current_stick += 1

for idx, numb in enumerate(counts):
    if numb in img_marker:
        print('Palito {}: {} pixels'.format(idx + 1, numb))


