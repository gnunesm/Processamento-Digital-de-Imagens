import cv2
# import gif2numpy
import numpy as np
from math import sqrt
import imageio
# from scipy import ndimage

gl = 0.5
gh = 2
d0 = 150
c = 1

def distance(u, v, p, q):
    return sqrt(((u+0.5) - p/2)**2 + ((v+0.5) - q/2)**2)

def homomorphic_filter(mag_img):
    hfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            hfilter[l][c] = (gh-gl)*(1-np.exp(-c*(distance(l, c, mag_img.shape[0], mag_img.shape[1])**2)/(d0**2)))+gl
    cv2.imwrite('filter.png', 255.0*hfilter)
    filtered_mag = np.multiply(mag_img, hfilter)
    return filtered_mag

# img = ndimage.imread('mar-il.gif')
# converted, _, _ = gif2numpy.convert('mar-il.gif')
# img = converted[0]
# new_img = np.zeros((img.shape[0], img.shape[1]))
# print(img.shape)
# for l_idx, l in enumerate(img):
#     for c_idx, c in enumerate(l):
#         # print(c[0])
#         new_img[l_idx][c_idx] = c[0]
#         # print(img[l_idx][c_idx], new_img[l_idx][c_idx])
# print(new_img.shape)
# print(new_img)
# cv2.imshow('Imagem', new_img)
# cv2.waitKey()
# cv2.imwrite('converted.png', new_img)
# cv2.imwrite('converted_rgb.png', img)
# new_img = new_img.astype(np.int64)

img = np.squeeze(np.array(imageio.mimread("mar-il.gif")))

# processo para filtragem homomórfica
z = np.log(img)
print(z.shape)
padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
padded_img[:img.shape[0], :img.shape[1]] = z
f = np.fft.fft2(padded_img)
fshift = np.fft.fftshift(f)
filtered_mag = homomorphic_filter(fshift)
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.exp(filtered_img)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('mar_il_filtered.png', filtered_img[:img.shape[0], :img.shape[1]])
