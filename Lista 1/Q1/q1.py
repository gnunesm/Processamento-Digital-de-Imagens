import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

img = cv2.imread('Fig8.02.jpg', 0)

pixel_values = [0]*256
for l in img:
    for c in l:
        pixel_values[c] += 1

plt.bar(np.arange(256), pixel_values)
plt.show()

thresh = 159

img[img > thresh] = 255
img[img <= thresh] = 0
cv2.imwrite('binary_sticks.png', img)

img_marker = np.zeros(img.shape)
counts = []

current_stick = 1

def neighbor_marking(mark, line, column):
    marked = 0
    for l in range(line - 1, line + 2):
        for c in range(column - 1, column + 2):
            if img[l][c] != 0:
                if img_marker[l][c] == 0:
                    img_marker[l][c] = mark
                    marked += 1
                    marked += neighbor_marking(mark, l, c)
    return marked

for m in range(img.shape[0]):
    for n in range(img.shape[1]):
        if img_marker[m][n] == 0:
            if img[m][n] != 0:
                img_marker[m][n] = current_stick
                counts.append(1+neighbor_marking(current_stick, m, n))
                current_stick += 1

for idx, numb in enumerate(counts):
    print('Palito {}: {} pixels'.format(idx + 1, numb))

img_new = np.zeros(img.shape)

for n in range(1, 14):
    img_new[img_marker == n] = 255/13*n

cv2.imwrite('colored_sticks.png', img_new)