import cv2
from morphological import *

img = cv2.imread('Fig11.10.jpg', 0)
img = binarize(img, 127)
skeleton = np.zeros(img.shape)
# histogram(img)
eroded = np.copy(img)
while not np.all(eroded == 0):
    opened = opening(eroded, 3)
    sk = eroded - opened
    skeleton[sk == 255] = 255
    eroded = erosion(eroded, 3)
    # cv2.imwrite('eroded.png', eroded)
    # cv2.imwrite('opened.png', opened)

cv2.imwrite('skeleton.png', skeleton)