import cv2
import matplotlib.pyplot as plt
import numpy as np

thresh = 20

# leitura da imagem
img = cv2.imread('pontos.bmp', 0)
height, width = img.shape

# mapeamento dos pontos
points = []
for l in range(height):
    for c in range(width):
        if img[l][c] > 127:
            points.append((c, height-1-l))
points.sort(key=lambda t: t[0]) # ordenação de acordo com coordenada x (esquerda pra direita)

# ordenação dos pontos em sentido horário
ordered = []
while points:
    ordered.append(points.pop(0))
    points.sort(key=lambda t: (t[0] - ordered[-1][0])**2 + (t[1] - ordered[-1][1])**2)

ordered = np.array(ordered)

start_point = ordered[0]
end_point = max(ordered, key=lambda t: t[0])

closed_list = [end_point]
open_list = [end_point, start_point]

while open_list:
    for n in range(ordered.shape[0]):
        if (ordered[n] == open_list[-1]).all():
            start_idx = n
        elif (ordered[n] == closed_list[-1]).all():
            end_idx = n
    if start_idx+1 < end_idx:
        d = np.absolute(np.cross(open_list[-1]-closed_list[-1],ordered[start_idx+1:end_idx]-closed_list[-1])/np.linalg.norm(open_list[-1]-closed_list[-1]))
        if max(d) >= thresh:
            idx = np.argmax(d) + start_idx + 1
            open_list.append(ordered[idx])
        else:
            closed_list.append(open_list.pop())
    else:
        closed_list.append(open_list.pop())
closed_list.pop() # impede de fechar o loop

for p in ordered:
    plt.plot(p[0], p[1], 'go')

for n in range(len(closed_list)-1):
    plt.plot((closed_list[n][0], closed_list[n+1][0]), (closed_list[n][1], closed_list[n+1][1]), 'r-')

ordered = np.append(ordered, [ordered[0]], axis=0)
ordered = np.delete(ordered, 0, axis=0)

start_point = max(ordered, key=lambda t: t[0])
end_point = min(ordered, key=lambda t: t[0])

closed_list = [end_point]
open_list = [end_point, start_point]

while open_list:
    for n in range(ordered.shape[0]):
        if (ordered[n] == open_list[-1]).all():
            start_idx = n
        elif (ordered[n] == closed_list[-1]).all():
            end_idx = n
    if start_idx+1 < end_idx:
        d = np.absolute(np.cross(open_list[-1]-closed_list[-1],ordered[start_idx+1:end_idx]-closed_list[-1])/np.linalg.norm(open_list[-1]-closed_list[-1]))
        if max(d) >= thresh:
            idx = np.argmax(d) + start_idx + 1
            open_list.append(ordered[idx])
        else:
            closed_list.append(open_list.pop())
    else:
        closed_list.append(open_list.pop())
closed_list.pop() # impede de fechar o loop

for n in range(len(closed_list)-1):
    plt.plot((closed_list[n][0], closed_list[n+1][0]), (closed_list[n][1], closed_list[n+1][1]), 'r-')

plt.show()
