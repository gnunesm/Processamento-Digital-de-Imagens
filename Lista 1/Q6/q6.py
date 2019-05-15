import cv2
import numpy as np
from math import sqrt

# def notch_filter():

def distance(u, v, p, q):
    return sqrt((u - p/2)**2 + (v - q/2)**2)

def butterworth_lowpass(mag_img, order, cutoff):
    bfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            bfilter[l][c] = 1/(1+((distance(l+1,c+1,mag_img.shape[0],mag_img.shape[1])/cutoff)**(2*order)))
    # for l in bfilter:
    #     print(l)
    cv2.imwrite('filter.png', np.abs(20*np.log(np.abs(bfilter))))
    filtered_mag = np.multiply(mag_img, bfilter)
    return filtered_mag

img = cv2.imread('camisa.jpg', 0)
img = img.astype(np.int64)
padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
padded_img[:img.shape[0], :img.shape[1]] = img
cv2.imwrite('padded.png', padded_img)
f = np.fft.fft2(padded_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
cv2.imwrite('magnitude.png', magnitude_spectrum)

# fshift[fshift.shape[0]/2 - 3:fshift.shape[0]/2 + 3, :] = 0

# fshift[fshift.shape[0]//2 - 3:fshift.shape[0]//2 + 3, :fshift.shape[1]//2 - 5] = 0
# fshift[fshift.shape[0]//2 - 3:fshift.shape[0]//2 + 3, fshift.shape[1]//2 + 5:] = 0

# fshift[225:525, :230] = 0
# fshift[225:525, 770:] = 0

fshift = butterworth_lowpass(fshift, 20, 270)

magnitude_spectrum = np.copy(fshift)
magnitude_spectrum[magnitude_spectrum!=0] = 20*np.log(np.abs(fshift[fshift!=0]))
cv2.imwrite('filtered_magnitude.png', np.abs(magnitude_spectrum))

filtered_f = np.fft.ifftshift(fshift)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('filtered.png', filtered_img[:img.shape[0], :img.shape[1]])