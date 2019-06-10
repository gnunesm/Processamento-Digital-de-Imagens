import cv2
import numpy as np
from math import sqrt
import imageio

def butterworth_highpass(mag_img, order, cutoff):
    bfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            bfilter[l][c] = 1/(1+((cutoff/(distance(l, c, mag_img.shape[0], mag_img.shape[1])))**(2*order)))
    cv2.imwrite('highfilter.png', 255*bfilter)
    filtered_mag = np.multiply(mag_img, bfilter)
    return filtered_mag

def butterworth_lowpass(mag_img, order, cutoff):
    bfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            bfilter[l][c] = 1/(1+((distance(l,c,mag_img.shape[0],mag_img.shape[1])/cutoff)**(2*order)))
    cv2.imwrite('lowfilter.png', 255*bfilter)
    filtered_mag = np.copy(mag_img)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            filtered_mag[l][c] = mag_img[l][c] * bfilter[l][c]
    return filtered_mag

gl = 0.1
gh = 2
d0 = 300
c = 1

def distance(u, v, p, q):
    return sqrt(((u+0.5) - p/2)**2 + ((v+0.5) - q/2)**2)

def homomorphic_filter(mag_img):
    hfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            hfilter[l][c] = (gh-gl)*(1-np.exp(-c*(distance(l, c, mag_img.shape[0], mag_img.shape[1])**2)/(d0**2)))+gl
    filtered_mag = np.multiply(mag_img, hfilter)
    return filtered_mag

img = np.squeeze(np.array(imageio.mimread("mar-il.gif")))
cv2.imwrite('mar-il.png', img)

# processo para filtragem homomórfica
padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
padded_img[:img.shape[0], :img.shape[1]] = img
f = np.fft.fft2(padded_img)
fshift = np.fft.fftshift(f)
filtered_mag = homomorphic_filter(fshift)
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('mar_il_filtered.png', filtered_img[:img.shape[0], :img.shape[1]])

# filtragem 2
low_filtered = butterworth_lowpass(fshift, 20, 1)
high_filtered = butterworth_highpass(fshift, 20, 20)
butterwort_filtered = low_filtered+high_filtered
magnitude_spectrum = np.abs(np.copy(filtered_mag))
magnitude_spectrum[magnitude_spectrum!=0] = 20*np.log(magnitude_spectrum[magnitude_spectrum!=0])
cv2.imwrite('filtered_magnitude.png', magnitude_spectrum)
filtered_f = np.fft.ifftshift(butterwort_filtered)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('butterworth_filtered.png', filtered_img[:img.shape[0], :img.shape[1]])