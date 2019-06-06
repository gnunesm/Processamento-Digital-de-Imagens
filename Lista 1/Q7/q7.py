import cv2
import numpy as np
from math import sqrt

def distance(u, v, p, q, uk, vk):
    return sqrt((u + 0.5 - p/2 - uk)**2 + (v + 0.5 - q/2 - vk)**2)

def butterworth_highpass(mag_img, order, cutoff, uk, vk):
    bfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            bfilter[l][c] = 1/(1+((cutoff/(distance(l, c, mag_img.shape[0], mag_img.shape[1], uk, vk)))**(2*order)))
    cv2.imwrite('filter.png', 255*bfilter)
    filtered_mag = np.multiply(mag_img, bfilter)
    return filtered_mag

def notch_filter(mag_img, order, cutoff, coords):
    for coord in coords:
        uk = coord[0]
        vk = coord[1]
        mag_img = butterworth_highpass(mag_img, order, cutoff, uk, vk)
        mag_img = butterworth_highpass(mag_img, order, cutoff, -uk, -vk)
    return mag_img

img = cv2.imread('mit_noise_periodic.jpg', 0)
img = img.astype(np.int64)
padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
padded_img[:img.shape[0], :img.shape[1]] = img
cv2.imwrite('padded.png', padded_img)
f = np.fft.fft2(padded_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
cv2.imwrite('magnitude.png', magnitude_spectrum)

x = 325
y = 240

coords = []

for _ in range(6):
    coords += [(y, x), (-y, x)]
    x += 130

x = 325
y = 240

for _ in range(6):
    y += 100
    coords += [(y, x), (-y, x)]

coords += [(0, 325+130), (0, -325-130)]
coords += [(340, 0), (-340, 0)]

filtered_mag = notch_filter(fshift, 20, 80, coords)

magnitude_spectrum = np.abs(np.copy(filtered_mag))
magnitude_spectrum[magnitude_spectrum!=0] = 20*np.log(magnitude_spectrum[magnitude_spectrum!=0])
cv2.imwrite('filtered_magnitude.png', magnitude_spectrum)
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('mit_filtered.png', filtered_img[:img.shape[0], :img.shape[1]])