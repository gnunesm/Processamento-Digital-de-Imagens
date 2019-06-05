import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram(img):
    pixel_values = [0]*256
    for l in img:
        for c in l:
            pixel_values[c] += 1
    plt.bar(np.arange(256), pixel_values)
    plt.show()

def binarize(img, thresh):
    img[img > thresh] = 255
    img[img <= thresh] = 0
    return img

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

img = cv2.imread('Fig11.10.jpg', 0)
# histogram(img)
img = binarize(img, 127)
eroded = erosion(img, 3)
cv2.imwrite('eroded.png', eroded)