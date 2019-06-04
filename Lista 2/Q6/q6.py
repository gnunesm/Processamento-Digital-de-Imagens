import cv2
import numpy as np
import matplotlib.pyplot as plt

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