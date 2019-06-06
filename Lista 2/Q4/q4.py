import cv2
import numpy as np

img = cv2.imread('pontos.bmp', 0)
(height, width) = img.shape

points = []
for l in range(height):
    for c in range(width):
        if img[l][c] > 127:
            points.append((c, height-1-l))
points.sort(key=lambda t: t[0])

ordered = []
while points:
    ordered.append(points.pop(0))
    # img[height-ordered[-1][1]-1][ordered[-1][0]] = 0
    # cv2.imwrite('img{}.png'.format(len(ordered)), img)
    points.sort(key=lambda t: (t[0] - ordered[-1][0])**2 + (t[1] - ordered[-1][1])**2)
ordered.append(ordered[0])

ordered = np.array(ordered)

start_point = ordered[0]
end_point = max(ordered, key=lambda t: t[0])

# ordered = np.delete(ordered, ordered.index())

closed_list = [end_point]
open_list = [end_point, start_point]

while open_list:
    start_idx = np.where(ordered == open_list[-1])
    end_idx = np.where(ordered == closed_list[-1])
    d = np.absolute(np.cross(open_list[-1]-closed_list[-1],ordered[start_idx+1:end_idx]-closed_list[-1])/np.linalg.norm(open_list[-1]-closed_list[-1]))
    if d:
        idx = np.argmax(d) + start_idx
        open_list.append(ordered[idx])
    else:
        closed_list.append(open_list[-1])
        np.delete(open_list, -1)

# descobrir se precisa usar o método do livro (se precisa ligar todos e se pode usar minha ordenação já feita)




    
