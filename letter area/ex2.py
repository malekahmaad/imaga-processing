import cv2
import numpy as np


def min_max(li):
    minX = li[0][1]
    minY = li[0][0]
    maxX = 0
    maxY = 0
    for element in li:
        if minX > element[1]:
            minX = element[1]
        
        if minY > element[0]:
            minY = element[0]

        if maxX < element[1]:
            maxX = element[1]

        if maxY < element[0]:
            maxY = element[0]

    return minX, minY, maxX, maxY


image = cv2.imread('baffalo.png',  cv2.IMREAD_GRAYSCALE)
red_image = cv2.imread('baffalo.png')
laplacian = cv2.Laplacian(image, cv2.CV_64F)
ret, th = cv2.threshold(laplacian, 35, 255, cv2.THRESH_BINARY)
visited = np.zeros_like(th, dtype=bool)

for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        if th[row, col] > 0 and visited[row, col] == False:
            object_list = []
            stack = [(row, col)]
            while stack:
                cur_row, cur_col = stack.pop()
                if th[cur_row, cur_col] > 0 and visited[cur_row, cur_col] == False:
                    visited[cur_row, cur_col] = True
                    object_list.append((cur_row, cur_col))
                    if 0 <= cur_col - 1:
                        stack.append((cur_row, cur_col - 1))
                    
                    if cur_col + 1 < image.shape[1]:
                        stack.append((cur_row, cur_col + 1))
    
                    if 0 <= cur_row - 1:
                        stack.append((cur_row - 1, cur_col))
                    
                    if cur_row + 1 < image.shape[0]:
                        stack.append((cur_row + 1, cur_col))
            
            minX, minY, maxX, maxY = min_max(object_list)
            if len(object_list) > 20:
                # print(minX, minY, maxX, maxY)
                # print(len(object_list))
                height = maxY - minY + 1
                width = maxX - minX + 1
                if 0.8*len(object_list) <= 2*(width + height) <= 1.5*len(object_list):
                    if 550 < width * height < 670:
                        print(f"the letter l area is equal to {width * height}")
                        # print(len(red_image[0]))
                        red_image[minY][minX] = [0, 0, 255]
                        red_image[minY][maxX] = [0, 0, 255]
                        red_image[maxY][minX] = [0, 0, 255]
                        red_image[maxY][maxX] = [0, 0, 255]

cv2.imshow('image with red points', red_image)
# cv2.imshow('threshold', th)
# cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()
