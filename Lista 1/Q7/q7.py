import cv2
import numpy as np
from math import sqrt

# def distance(u, v, p, q, uk, vk):
#     return sqrt((u + 0.5 - p/2 - uk)**2 + (v + 0.5 - q/2 - vk)**2)

# def butterworth_highpass(mag_img, order, cutoff, uk, vk):
#     bfilter =  np.zeros(mag_img.shape, np.float64)
#     for l in range(mag_img.shape[0]):
#         for c in range(mag_img.shape[1]):
#             bfilter[l][c] = 1/(1+((cutoff/(distance(l, c, mag_img.shape[0], mag_img.shape[1], uk, vk)))**(2*order)))
#     cv2.imwrite('filter.png', 255*bfilter)
#     filtered_mag = np.multiply(mag_img, bfilter)
#     return filtered_mag

# def notch_filter(mag_img, order, cutoff, coords):
#     for coord in coords:
#         uk = coord[0]
#         vk = coord[1]
#         mag_img = butterworth_highpass(mag_img, order, cutoff, uk, vk)
#         mag_img = butterworth_highpass(mag_img, order, cutoff, -uk, -vk)
#     return mag_img

# img = cv2.imread('mit_noise_periodic.jpg', 0)
# img = img.astype(np.int64)
# padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
# padded_img[:img.shape[0], :img.shape[1]] = img
# cv2.imwrite('padded.png', padded_img)
# f = np.fft.fft2(padded_img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# cv2.imwrite('magnitude.png', magnitude_spectrum)

# x = 320
# y = 240

# coords = ((y, x), (-y, x))
# filtered_mag = notch_filter(fshift, 5, 40, coords)

# magnitude_spectrum = np.abs(np.copy(filtered_mag))
# magnitude_spectrum[magnitude_spectrum!=0] = 20*np.log(magnitude_spectrum[magnitude_spectrum!=0])
# cv2.imwrite('filtered_magnitude.png', magnitude_spectrum)
# filtered_f = np.fft.ifftshift(filtered_mag)
# filtered_img = np.fft.ifft2(filtered_f)
# filtered_img = np.abs(filtered_img)
# filtered_img = filtered_img.astype(np.uint8)
# cv2.imwrite('mit_filtered.png', filtered_img[:img.shape[0], :img.shape[1]])

def distance(u, v, p, q):
    return sqrt(((u+0.5) - p/2)**2 + ((v+0.5) - q/2)**2)

def butterworth_lowpass(mag_img, order, cutoff):
    bfilter =  np.zeros(mag_img.shape, np.float64)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            bfilter[l][c] = 1/(1+((distance(l,c,mag_img.shape[0],mag_img.shape[1])/cutoff)**(2*order)))
    print(bfilter.shape)
    cv2.imwrite('filter_n{}_D{}.png'.format(order, cutoff), 255.0*bfilter)
    # for l in 255*bfilter:
    #     if 255 in l:
    #         for c in l:
    #             print(c)
    # filtered_mag = np.multiply(mag_img, bfilter)
    filtered_mag = np.copy(mag_img)
    for l in range(mag_img.shape[0]):
        for c in range(mag_img.shape[1]):
            filtered_mag[l][c] = mag_img[l][c] * bfilter[l][c]
    # print(bfilter)
    return filtered_mag

img = cv2.imread('lena.tif', 0)
img = img.astype(np.int64)
padded_img = np.zeros((2*img.shape[0], 2*img.shape[1]), np.int64)
padded_img[:img.shape[0], :img.shape[1]] = img
cv2.imwrite('padded.png', padded_img)
f = np.fft.fft2(padded_img)
fshift = np.fft.fftshift(f)
print(fshift.dtype)
magnitude_spectrum = 20*np.log(np.abs(fshift))
cv2.imwrite('padded_magnitude.png', magnitude_spectrum)
# cv2.imwrite('padded_magnitude.png', np.abs(fshift))

n = 1
d = 20

filtered_mag = butterworth_lowpass(fshift, n, d)
print(filtered_mag.dtype)
magnitude_spectrum = np.abs(np.copy(filtered_mag))
magnitude_spectrum[magnitude_spectrum!=0] = 20*np.log(magnitude_spectrum[magnitude_spectrum!=0])
print(magnitude_spectrum.dtype)
cv2.imwrite('filtered_magnitude_n{}_d{}.png'.format(n, d), magnitude_spectrum)
filtered_f = np.fft.ifftshift(filtered_mag)
filtered_img = np.fft.ifft2(filtered_f)
filtered_img = np.abs(filtered_img)
filtered_img = filtered_img.astype(np.uint8)
cv2.imwrite('filtered_n{}_{}.png'.format(n, d), filtered_img[:img.shape[0], :img.shape[1]])