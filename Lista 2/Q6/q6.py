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

def region_grow(l, c):
    for m in range(l-1, l+2):
        for n in range(c-1, c+2):
            if eroded[m][n] != 255 and diff_img[m][n] <= t:
                eroded[m][n] = 255
                region_grow(m, n)

img = cv2.imread('Fig10.40(a).jpg', 0)

# pixel_values = [0]*256
# for l in img:
#     for c in l:
#         pixel_values[c] += 1

# plt.bar(np.arange(256), pixel_values)
# plt.show()
original_img = np.copy(img)
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
eroded[eroded!=0] = 255*np.ones(eroded[eroded!=0].shape)
cv2.imwrite('eroded.png', eroded)

t = 68

diff_img = 255 - original_img
cv2.imwrite('diff_img.png', diff_img)

for l in range(eroded.shape[0]):
    for c in range(eroded.shape[1]):
        if eroded[l][c] == 255:
            region_grow(l, c)

cv2.imwrite('final.png', eroded)

cv2.imwrite('diff.png', eroded - img)