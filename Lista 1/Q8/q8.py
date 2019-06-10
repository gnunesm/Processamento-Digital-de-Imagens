import cv2
import math
import numpy as np
import os

def filtering(mask, img):
    total = 0
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            total += mask[m][n] * img[m][n]
    return total

def convolution_lowpass(n, img):
    extension = n - 1
    mask = np.ones((n, n), dtype=np.int64)
    extended_img = np.zeros((img.shape[0]+extension, img.shape[1]+extension), dtype=np.int64)
    extended_img[extension//2:-extension//2, extension//2:-extension//2] = img
    filtered_img = np.zeros(img.shape, dtype=np.int64)
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            filtered_img[l][c] = filtering(mask, extended_img[l:l+n, c:c+n])
    filtered_img = filtered_img/(n*n)
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

def mse(original_img, filtered_img):
    m = original_img.shape[0]
    n = original_img.shape[1]
    total = 0
    for l in range(m):
        for c in range(n):
            total += (original_img[l][c] - filtered_img[l][c])**2
    print(total/(m*n))
    return  total/(m*n)

def psnr(original_img, filtered_img):
    return 20*math.log10(255/(math.sqrt(mse(original_img, filtered_img))))

def median_filter(mask, img):
    values = []
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            values.append(img[m][n])
    values.sort()
    return values[len(values)//2]

def median(img, n=11):
    extension = n - 1
    mask = np.ones((n, n), dtype=np.int64)
    extended_img = np.zeros((img.shape[0]+extension, img.shape[1]+extension), dtype=np.int64)
    extended_img[extension//2:-extension//2, extension//2:-extension//2] = img
    filtered_img = np.zeros(img.shape, dtype=np.int64)
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            filtered_img[l][c] = median_filter(mask, extended_img[l:l+n, c:c+n])
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

def etapa_B(block):
    line = block.reshape((1, -1))[0]
    zxy = line[len(line)//2]
    line.sort()
    zmed = line[len(line)//2]
    zmin = line[0]
    zmax = line[-1]
    B1 =  zxy - zmin
    B2 = zxy - zmax
    if B1 > 0 and B2 < 0:
        return zxy
    else:
        return zmed

def etapa_A(block):
    line = block.reshape((1, -1))[0]
    line.sort()
    zmed = line[len(line)//2]
    zmin = line[0]
    zmax = line[-1]
    A1 = zmed - zmin
    A2 = zmed - zmax
    if A1 > 0 and A2 < 0:
        return (True, etapa_B(block))
    else:
        return (False, zmed)

def adapt_median(img, n=11):
    extension = n - 1
    extended_img = np.zeros((img.shape[0]+extension, img.shape[1]+extension), dtype=np.int64)
    extended_img[extension//2:-extension//2, extension//2:-extension//2] = img
    filtered_img = np.zeros(img.shape, dtype=np.int64)
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            curr_n = 3
            ok = False
            while curr_n <= n and not ok:
                ok, filtered_img[l][c] = etapa_A(extended_img[l+extension//2-curr_n//2:l+extension//2+curr_n//2, c+extension//2-curr_n//2:c+extension//2+curr_n//2])
                curr_n += 2
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

img = cv2.imread('ruidosa1.tif', 0)
img = img.astype(np.int64)
original_img = cv2.imread('original.tif', 0)
original_img = original_img.astype(np.int64)
print('Filtros sobre ruidosa1')
if not os.path.isfile('media1.png'):
    med_filtered = convolution_lowpass(11, img)
    cv2.imwrite('media1.png', med_filtered)
    print('Filtro de média')
    print('PSNR =', psnr(original_img, med_filtered))

if not os.path.isfile('mediana1.png'):
    median_filtered = median(img)
    cv2.imwrite('mediana1.png', median_filtered)
    print('Filtro de mediana')
    print('PSNR =', psnr(original_img, median_filtered))

if not os.path.isfile('adapt_mediana1.png'):
    adapt_median_filtered = adapt_median(img)
    cv2.imwrite('adapt_mediana1.png', adapt_median_filtered)
    print('Filtro de mediana adaptativo')
    print('PSNR =', psnr(original_img, adapt_median_filtered))

print('PSNR com ruidosa1 =', psnr(original_img, img))

img = cv2.imread('ruidosa2.tif', 0)
img = img.astype(np.int64)
print('Filtros sobre ruidosa2')
if not os.path.isfile('media2.png'):
    med_filtered = convolution_lowpass(11, img)
    cv2.imwrite('media2.png', med_filtered)
    print('Filtro de média')
    print('PSNR =', psnr(original_img, med_filtered))

if not os.path.isfile('mediana2.png'):
    median_filtered = median(img)
    cv2.imwrite('mediana2.png', median_filtered)
    print('Filtro de mediana')
    print('PSNR =', psnr(original_img, median_filtered))

if not os.path.isfile('adapt_mediana2.png'):
    adapt_median_filtered = adapt_median(img)
    cv2.imwrite('adapt_mediana2.png', adapt_median_filtered)
    print('Filtro de mediana adaptativo')
    print('PSNR =', psnr(original_img, adapt_median_filtered))

print('PSNR com ruidosa2 =', psnr(original_img, img))