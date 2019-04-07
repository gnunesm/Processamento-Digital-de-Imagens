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
    mask = np.ones((n, n), dtype=np.int64)
    extended_img = np.zeros((img.shape[0]+extension, img.shape[1]+extension), dtype=np.int64)
    extended_img[extension//2:-extension//2, extension//2:-extension//2] = img
    filtered_img = np.zeros(img.shape, dtype=np.int64)
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            filtered_img[l][c] = filtering(mask, extended_img[l:l+n, c:c+n])
    filtered_img = filtered_img/(n*n)
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

def convolution_laplacian(n, img):
    extension = n - 1
    mask = np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]], np.int64)
    # mask = mask * -1
    extended_img = np.zeros((img.shape[0]+extension, img.shape[1]+extension), dtype=np.int64)
    extended_img[extension//2:-extension//2, extension//2:-extension//2] = img
    filtered_img = np.zeros(img.shape, dtype=np.int64)
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            filtered_img[l][c] = filtering(mask, extended_img[l:l+n, c:c+n])
    filtered_img = filtered_img/(n*n)
    filtered_img = img + filtered_img
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

img = cv2.imread('lena.tif', 0)
img = img.astype(np.int64)
for n in range(3, 17, 2):
    # filtered_img = convolution_lowpass(n, img)
    # cv2.imwrite('lowpass_{}.png'.format(n), filtered_img)

n = 3
filtered_img = convolution_laplacian(n, img)
cv2.imwrite('laplacian_{}.png'.format(n), filtered_img)