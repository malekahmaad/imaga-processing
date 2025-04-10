import cv2
import numpy as np


image = cv2.imread('baffalo.png', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

for i in range(len(image)):
    for j in range(int(len(image[0])/2)+75):
        laplacian[i][j] = 0

for i in range(len(image)):
    for j in range(int(len(image[0])/2)+95, len(image[0])):
        laplacian[i][j] = 0

for i in range(int(len(image)/2)+30):
    for j in range(int(len(image[0])/2)+73, int(len(image[0])/2)+95):
        laplacian[i][j] = 0

for i in range(len(image)-20, len(image)):
    for j in range(int(len(image[0])/2)+74, int(len(image[0])/2)+95):
        laplacian[i][j] = 0

ret, th = cv2.threshold(laplacian, 50, 255, cv2.THRESH_BINARY)

for j in range(int(len(image[0])/2)+74, int(len(image[0])/2)+95):
    th[int(len(image)/2)+30][j] = 0
    th[len(image)-21][j] = 0

for i in range(int(len(image)/2)+30, len(image)-21):
    th[i][int(len(image[0])/2)+74] = 0
    th[i][int(len(image[0])/2)+94] = 0

x1 = 0
x2 = 0
y1 = 0
y2 = 0

whites = np.argwhere(th == 255)

y1 = whites[0][0]
x1 = whites[0][1]
y2 = whites[len(whites)-1][0]
x2 = whites[len(whites)-1][1]

print(x1, x2, y1, y2)

print(f"Area is equal to: {(x2-x1+1)*(y2-y1+1)}")

red_image = cv2.imread('baffalo.png')

red_image[y1][x1] = [0, 0, 255]
red_image[y1][x2] = [0, 0, 255]
red_image[y2][x1] = [0, 0, 255]
red_image[y2][x2] = [0, 0, 255]

cv2.imshow('Original', red_image)
# cv2.imshow('Original2', th)
# cv2.imshow('Laplacian Edge Detection2', laplacian)
cv2.waitKey(0)
