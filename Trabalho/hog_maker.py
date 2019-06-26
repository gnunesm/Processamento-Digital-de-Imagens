import cv2
import numpy as np

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

# def calc_l2(vectors):

img = cv2.imread('crop001053.png')
# img = img.astype(np.float64)

h_grad = np.copy(img)
v_grad = np.copy(img)

for m in range(img.shape[0]):
    for n in range(img.shape[1]):
        if m == 0:
            v_grad[m][n] = img[m+1][n] - img[m][n]
        elif m == img.shape[0] - 1:
            v_grad[m][n] = img[m][n] - img[m-1][n]
        else:
            v_grad[m][n] = img[m+1][n] - img[m-1][n]
        if n == 0:
            h_grad[m][n] = img[m][n+1] - img[m][n]
        elif n == img.shape[1] - 1:
            h_grad[m][n] = img[m][n] - img[m][n-1]
        else:
            h_grad[m][n] = img[m][n+1] - img[m][n-1]

grad_mag = np.sqrt(np.square(h_grad) + np.square(v_grad))
grad_dir = np.arctan(v_grad/h_grad)
grad_dir[np.isnan(grad_dir)] = np.pi/2
grad_dir = np.degrees(grad_dir)
grad_dir[grad_dir<0] += 180
grad_dir = np.absolute(grad_dir)

print(grad_mag.shape)

final_mag = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
final_dir = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
idxs = np.argmax(grad_mag, axis=2)

for m in range(final_mag.shape[0]):
    for n in range(final_mag.shape[1]):
        final_mag[m][n] = grad_mag[m][n][idxs[m][n]]
        final_dir[m][n] = grad_dir[m][n][idxs[m][n]]

cell_size = 8
h_cells = img.shape[1]//cell_size
v_cells = img.shape[0]//cell_size

cells = []

for m in range(v_cells):
    cells.append([])
    for n in range(h_cells):
        hist = [0,0,0,0,0,0,0,0,0]
        for l in range(cell_size*m, cell_size*(m+1)):
            for c in range(cell_size*n, cell_size*(n+1)):
                angle = final_dir[l][c]
                if angle == 180:
                    hist[8] += final_mag[l][c]
                else:
                    hist[int(angle//20)] += final_mag[l][c]
        cells[-1].append(hist)

cells = np.array(cells)
print(cells.shape)
print(cells[0][0])

# descriptor = []

# for l in range(cells.shape[0]-1):
#     for c in range(cells.shape[1]-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)

# img = img.astype(np.int8)

ax1.axis('off')
ax1.imshow(img)
ax1.set_title('Input image')

ax2.axis('off')
ax2.bar(np.arange(9), cells[0][0])
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
# plt.bar(np.arange(256), pixel_values)