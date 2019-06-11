import cv2
import imutils
import numpy as np
from tabulate import tabulate

def m(img, p, q):
    return np.sum((x**p)*(y**q)*img)

def x_b(img):
    return m(img, 1, 0)/m(img, 0, 0)

def y_b(img):
    return m(img, 0, 1)/m(img, 0, 0)

def mi(img, p, q):
    return np.sum(((x - x_b(img))**p)*((y - y_b(img))**q)*img)

def eta(img, p, q):
    return mi(img, p, q)/(mi(img, 0, 0)**((p+q)/2+1))

def phi1(img):
    return eta(img, 2, 0) + eta(img, 0, 2)

def phi2(img):
    return (eta(img, 2, 0) - eta(img, 0, 2))**2 + 4*(eta(img, 1, 1)**2)

def phi3(img):
    return (eta(img, 3, 0) - 3*eta(img, 1, 2))**2 + (3*eta(img, 2, 1) - eta(img, 0, 3))**2

def phi4(img):
    return (eta(img, 3, 0) + eta(img, 1, 2))**2 + (eta(img, 2, 1) + eta(img, 0, 3))**2

def phi5(img):
    return (eta(img, 3, 0) - 3*eta(img, 1, 2))*(eta(img, 3, 0) + eta(img, 1, 2))*((eta(img, 3, 0)+eta(img, 1, 2))**2-3*((eta(img, 2, 1)+eta(img, 0, 3))**2))+(3*eta(img, 2, 1)-eta(img, 0,3))*(eta(img,2,1)+eta(img,0,3))*(3*((eta(img,3,0)+eta(img,1,2))**2)-(eta(img,2,1)+eta(img,0,3))**2)

def phi6(img):
    return (eta(img,2,0)-eta(img,0,2))*((eta(img,3,0)+eta(img,1,2))**2-(eta(img,2,1)+eta(img,0,3))**2)+4*eta(img,1,1)*(eta(img,3,0)+eta(img,1,2))*(eta(img,2,1)+eta(img,0,3))

def phi7(img):
    return (3*eta(img,2,1)-eta(img,0,3))*(eta(img,3,0)+eta(img,1,2))*((eta(img,3,0)+eta(img,1,2))**2-3*((eta(img,2,1)+eta(img,0,3))**2))+(3*eta(img,1,2)-eta(img,3,0))*(eta(img,2,1)+eta(img,0,3))*(3*((eta(img,3,0)+eta(img,1,2))**2)-(eta(img,2,1)+eta(img,0,3))**2)

img = cv2.imread('lena.tif', 0)
img = img.astype(np.float64)

x = np.array([np.arange(img.shape[0]), ]*img.shape[1]).transpose()
y = np.array([np.arange(img.shape[1]), ]*img.shape[0])

phi = [[], [], [], [], [], [], []]

phi[0].append(phi1(img))
phi[1].append(phi2(img))
phi[2].append(phi3(img))
phi[3].append(phi4(img))
phi[4].append(phi5(img))
phi[5].append(phi6(img))
phi[6].append(phi7(img))

new_img = np.zeros(img.shape, dtype = np.float64)
new_img[img.shape[0]//4:3*img.shape[0]//4, img.shape[1]//4:3*img.shape[1]//4] = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2))
cv2.imwrite('resized.png', new_img)

phi[0].append(phi1(new_img))
phi[1].append(phi2(new_img))
phi[2].append(phi3(new_img))
phi[3].append(phi4(new_img))
phi[4].append(phi5(new_img))
phi[5].append(phi6(new_img))
phi[6].append(phi7(new_img))

new_img = imutils.rotate_bound(img, 90)
cv2.imwrite('rot90.png', new_img)

phi[0].append(phi1(new_img))
phi[1].append(phi2(new_img))
phi[2].append(phi3(new_img))
phi[3].append(phi4(new_img))
phi[4].append(phi5(new_img))
phi[5].append(phi6(new_img))
phi[6].append(phi7(new_img))

new_img = imutils.rotate_bound(img, 180)
cv2.imwrite('rot180.png', new_img)

phi[0].append(phi1(new_img))
phi[1].append(phi2(new_img))
phi[2].append(phi3(new_img))
phi[3].append(phi4(new_img))
phi[4].append(phi5(new_img))
phi[5].append(phi6(new_img))
phi[6].append(phi7(new_img))

phi = np.sign(phi)*np.absolute(np.log10(np.absolute(phi)))
headers = ['Original', 'Resized', 'Rotated 90', 'Rotated 180']
print(tabulate(phi, headers=headers))