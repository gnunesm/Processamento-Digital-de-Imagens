import cv2
import numpy as np

# delta = np.pi/6
delta = (22, 10)

img = cv2.imread('peppers.tiff')
# img = img.astype(np.float64)
# print(img.shape)

# hsi_img = np.zeros(img.shape, np.float64)

# theta = np.arccos(0.5*(img[:,:,2]-img[:,:,1]+img[:,:,2]-img[:,:,0])/(np.sqrt((img[:,:,2]-img[:,:,1])**2+(img[:,:,2]-img[:,:,0])*(img[:,:,1]-img[:,:,0]))))

# for l in range(img.shape[0]):
#     for c in range(img.shape[1]):
#         if img[l][c][0] <= img[l][c][2]:
#             hsi_img[l][c][0] = theta[l][c]
#         else:
#             hsi_img[l][c][0] = 2*np.pi - theta[l][c]
#         hsi_img[l][c][1] = 1 - 3*img[l][c].min()/(img[l][c][2]+img[l][c][1]+img[l][c][0])

# hsi_img[:,:,2] = (img[:,:,2] + img[:,:,1] + img[:,:,0])/3

# hsi_img[:,:,0][hsi_img[:,:,0]<=delta] += 4*np.pi/3
# hsi_img[:,:,0][hsi_img[:,:,0]>=2*np.pi-delta] -= 2*np.pi/3

# img[:,:,0] = hsi_img[:,:,2]*(1 - hsi_img[:,:,1])

hsi_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsi_img.shape)
hsi_img[:,:,0][hsi_img[:,:,0]<=delta[0]] += 120
hsi_img[:,:,0][hsi_img[:,:,0]>=180-delta[1]] -= 60
img = cv2.cvtColor(hsi_img, cv2.COLOR_HSV2BGR)

cv2.imwrite('changed.png', img)


