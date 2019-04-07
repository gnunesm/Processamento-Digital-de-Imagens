import cv2
import numpy as np

def filtering(mask, img):
    total = 0
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            total += mask[m][n] * img[m][n]
    return total

def convolution_lowpass(n, img):
    extension = n - 1
    mask = np.ones((n, n), dtype=np.uint8)
    extended_img = np.zeros((img.shape[0]+extension, img.shape[1]+extension), dtype=np.uint8)
    extended_img[extension//2:-extension//2, extension//2:-extension//2] = img
    filtered_img = np.zeros(img.shape, dtype=np.uint16)
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            filtered_img[l][c] = filtering(mask, extended_img[l:l+n, c:c+n])
    print(filtered_img)
    filtered_img = filtered_img/(n*n)
    print(filtered_img)
    return filtered_img

n = 11
img = cv2.imread('lena.tif', 0)
filtered_img = convolution(n, img)
cv2.imwrite('filtered_{}.png'.format(n), filtered_img)
