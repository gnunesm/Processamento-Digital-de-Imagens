import cv2
import numpy as np
import matplotlib.pyplot as plt

def mod_erosion(img, n):
    height = img.shape[0]
    width = img.shape[1]
    eroded = np.copy(img)
    pad = n//2
    for l in range(pad, height-pad):
        for c in range(pad, width-pad):
            if np.any(img[l-pad:l+pad+1, c-pad:c+pad+1]==0):
                if img[l][c] != 0 and (eroded == img[l][c]).sum() > 1:
                    eroded[l][c] = 0
                elif  img[l][c] == 0:
                    eroded[l][c] = 0
    return eroded

def neighbor_marking(mark, line, column):
    for l in range(line - 1, line + 2):
        for c in range(column - 1, column + 2):
            if img[l][c] != 0:
                if img_marker[l][c] == 0:
                    img_marker[l][c] = mark
                    neighbor_marking(mark, l, c)

img = cv2.imread('Fig10.40(a).jpg', 0)

# pixel_values = [0]*256
# for l in img:
#     for c in l:
#         pixel_values[c] += 1

# plt.bar(np.arange(256), pixel_values)
# plt.show()

thresh = 254

img[img > thresh] = 255
img[img <= thresh] = 0
cv2.imwrite('3-binary.png', img)

img_marker = np.zeros(img.shape)
component_number = 1
for m in range(img.shape[0]):
    for n in range(img.shape[1]):
        if img_marker[m][n] == 0:
            if img[m][n] != 0:
                img_marker[m][n] = component_number
                neighbor_marking(component_number, m, n)
                component_number += 1
                print(component_number)

cv2.imwrite('marked.png', 15*img_marker)
eroded = img_marker
for _ in range(10):
    eroded = mod_erosion(eroded, 3)
cv2.imwrite('eroded.png', eroded)