import cv2
import numpy as np

img = np.zeros((512,512))

img[:, 256] = np.array([255]*512)

cv2.imwrite('line.png', img)

padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
padded_img[:img.shape[0], :img.shape[1]] = img
f = np.fft.fft2(padded_img)
fshift = np.fft.fftshift(f)

magnitude_spectrum = np.abs(fshift)
magnitude_spectrum[magnitude_spectrum!=0] = 20*np.log(magnitude_spectrum[magnitude_spectrum!=0])
cv2.imwrite('magnitude.png', magnitude_spectrum)