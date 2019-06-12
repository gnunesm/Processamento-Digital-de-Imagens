import cv2
import numpy as np
import matplotlib.pyplot as plt

def mod_erosion(img, n):
    height = img.shape[0]
    width = img.shape[1]
    eroded = np.zeros(img.shape)
    pad = n//2
    eroded[pad:-1*pad, pad:-1*pad] = 255*np.ones([img.shape[0]-2*(pad), img.shape[1]-2*(pad)])
    for l in range(pad, height-pad):
        for c in range(pad, width-pad):
            if not np.all(img[l-pad:l+pad+1, c-pad:c+pad+1]==255):
                eroded[l][c] = 0
    return eroded

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

img = mod_erosion(img, 3)
cv2.imwrite('eroded.png', img)