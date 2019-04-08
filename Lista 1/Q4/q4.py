import cv2
import numpy as np

def distance(u, v, p, q):


def butterworth_lowpass(mag_img, order, cutoff):
    filtered_mag =  np.zeros(mag_img.shape)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            filtered_mag[l][c] = 1/(1+((distance(l+1,c+1,mag_img.shape[0],mag_img.shape[1])/cutoff)**(2*order)))
    return filtered_mag

img = cv2.imread('lena.tif', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
cv2.imwrite('magnitude.png', magnitude_spectrum)