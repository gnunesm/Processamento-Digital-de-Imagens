import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize(img, thresh):
    img[img > thresh] = 255
    img[img <= thresh] = 0
    return img

def histogram(img):
    pixel_values = [0]*256
    for l in img:
        for c in l:
            pixel_values[c] += 1
    plt.bar(np.arange(256), pixel_values)
    plt.show()

# supondo que n ímpar é o lado de um elemento estruturante quadrado
def erosion(img, n):
    height = img.shape[0]
    width = img.shape[1]
    eroded = np.zeros(img.shape)
    pad = n//2
    for l in range(pad, height-pad):
        for c in range(pad, width-pad):
            if np.all(img[l-pad:l+pad+1, c-pad:c+pad+1]==255):
                eroded[l][c] = 255
    return eroded

def opening(img, n):
    height = img.shape[0]
    width = img.shape[1]
    opened = np.zeros(img.shape)
    for l in range(height - n + 1):
        for c in range(width - n + 1):
            if np.all(img[l:l+n, c:c+n]==255):
                opened[l:l+n, c:c+n] = 255*np.ones((n, n))
    return opened

img = cv2.imread('Fig11.10.jpg', 0)
img = binarize(img, 127)
skeleton = np.zeros(img.shape)
eroded = np.copy(img)
while not np.all(eroded == 0):
    opened = opening(eroded, 3)
    sk = eroded - opened
    skeleton[sk == 255] = 255
    eroded = erosion(eroded, 3)

cv2.imwrite('skeleton.png', skeleton)