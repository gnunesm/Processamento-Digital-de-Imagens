import cv2
from morphological import *

img = cv2.imread('Fig11.10.jpg', 0)
img = binarize(img, 127)

eroded = erosion(img, 3)
img = img - eroded

cv2.imwrite('boundary.png', img)
resampled = img
for n in range(10):
    tmp = resampled
    resampled = np.zeros((resampled.shape[0]//2, resampled.shape[1]//2))
    for l in range(resampled.shape[0]):
        for c in range(resampled.shape[1]):
            if np.any(tmp[2*l:2*(l+1), 2*c:2*(c+1)] == 255):
                resampled[l][c] = 255
    cv2.imwrite('resampled{}.png'.format(n), resampled)