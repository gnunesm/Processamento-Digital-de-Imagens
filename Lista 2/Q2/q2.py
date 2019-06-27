import cv2
import math
import numpy as np

delta = (math.radians(50), math.radians(5))

img = cv2.imread('peppers.tiff')
cv2.imwrite('peppers.png', img)
img = img.astype(np.float64)

# RGB to HSI
hsi_img = np.zeros(img.shape, np.float64)
theta = np.zeros(img.shape[:2], np.float64)

for l in range(theta.shape[0]):
    for c in range(theta.shape[1]):
        if img[l][c].sum() != 0:
            theta[l][c] = np.arccos(0.5*(img[l,c,2]-img[l,c,1]+img[l,c,2]-img[l,c,0])/np.sqrt((img[l,c,2]-img[l,c,1])**2+(img[l,c,2]-img[l,c,0])*(img[l,c,1]-img[l,c,0])))

for l in range(img.shape[0]):
    for c in range(img.shape[1]):
        if img[l][c].sum() != 0:
            if img[l][c][0] <= img[l][c][2]:
                hsi_img[l][c][0] = theta[l][c]
            else:
                hsi_img[l][c][0] = 2*np.pi - theta[l][c]            
            hsi_img[l][c][1] = 1 - 3*img[l][c].min()/(img[l][c][2]+img[l][c][1]+img[l][c][0])

hsi_img[:,:,2] = (img[:,:,2] + img[:,:,1] + img[:,:,0])/3

# manipulation
hsi_img[:,:,0][hsi_img[:,:,0]<=delta[0]] += 4*np.pi/3
hsi_img[:,:,0][hsi_img[:,:,0]>=2*np.pi-delta[1]] -= 2*np.pi/3

# HSI to RGB
for l in range(img.shape[0]):
    for c in range(img.shape[1]):
        h = hsi_img[l][c][0]
        s = hsi_img[l][c][1]
        i = hsi_img[l][c][2] 
        if h < 2*np.pi/3:
            img[l][c][0] = i*(1 - s)
            img[l][c][2] = i*(1 + s*np.cos(h)/np.cos(np.pi/3-h))
            img[l][c][1] = 3*i - (img[l][c][2] + img[l][c][0])
        elif h >= 240:
            img[l][c][1] = i*(1 - s)
            img[l][c][0] = i*(1 + s*np.cos(h)/np.cos(np.pi/3-h))
            img[l][c][2] = 3*i - (img[l][c][1] + img[l][c][0])
        else:
            img[l][c][2] = i*(1 - s)
            img[l][c][1] = i*(1 + s*np.cos(h)/np.cos(np.pi/3-h))
            img[l][c][0] = 3*i - (img[l][c][2] + img[l][c][1])

cv2.imwrite('changed.png', img)


