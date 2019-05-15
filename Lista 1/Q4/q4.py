import cv2
import numpy as np
from math import sqrt

def distance(u, v, p, q):
    return sqrt((u - p/2)**2 + (v - q/2)**2)

def butterworth_lowpass(mag_img, order, cutoff):
    bfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            bfilter[l][c] = 1/(1+((distance(l+1,c+1,mag_img.shape[0],mag_img.shape[1])/cutoff)**(2*order)))
    print(bfilter.shape)
    for l in bfilter:
        print(l)
    filtered_mag = np.multiply(mag_img, bfilter)
    return filtered_mag

img = cv2.imread('lena.tif', 0)
img = img.astype(np.int64)
padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
padded_img[:img.shape[0], :img.shape[1]] = img
cv2.imwrite('padded.png', padded_img)
f = np.fft.fft2(padded_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
cv2.imwrite('padded_magnitude.png', magnitude_spectrum)
# cv2.imwrite('padded_magnitude.png', np.abs(fshift))

n = 1
d = 20

filtered_mag = butterworth_lowpass(fshift, n, d)
magnitude_spectrum = np.copy(filtered_mag)
magnitude_spectrum[magnitude_spectrum!=0] = 20*np.log(np.abs(filtered_mag[filtered_mag!=0]))
cv2.imwrite('filtered_magnitude.png', np.abs(magnitude_spectrum))
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('filtered_n{}_{}.png'.format(n, d), filtered_img[:img.shape[0], :img.shape[1]])

d = 80
filtered_mag = butterworth_lowpass(fshift, n, d)
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('filtered_n{}_{}.png'.format(n, d), filtered_img[:img.shape[0], :img.shape[1]])

n = 20
d = 20
filtered_mag = butterworth_lowpass(fshift, n, d)
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('filtered_n{}_{}.png'.format(n, d), filtered_img[:img.shape[0], :img.shape[1]])

d = 80
filtered_mag = butterworth_lowpass(fshift, n, d)
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('filtered_n{}_{}.png'.format(n, d), filtered_img[:img.shape[0], :img.shape[1]])

# f = np.fft.ifftshift(fshift)
# original_img = np.fft.ifft2(f)
# original_img = np.abs(original_img)
# cv2.imwrite('remade.png', original_img)