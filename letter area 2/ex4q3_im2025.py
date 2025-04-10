"""
malek ahmad
324921345

first i did erode so i make all the objects smaller then i subtract the original image with the erode one and by doing this i get the borders of every object
then for the second part of the question i did the region filling algorithm that we took in class using the erode image
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("rice-shaded.tif", cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
cl1 = clahe.apply(img)
ret, th = cv2.threshold(cl1, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
erode = cv2.erode(th, kernel, iterations=1)

borders = th - erode
rows, cols = th.shape
# filled = np.copy(borders)
filled = np.zeros_like(img)
filled[rows-1, cols-1] = 255
while True:
    # print("entered")
    prev_filled = filled.copy()
    filled = cv2.dilate(prev_filled, kernel, iterations=1)
    filled = filled & ~borders
    if np.array_equal(filled, prev_filled):
        break

# print(filled)
final = filled | borders

for i in range(len(final)):
    for j in range(len(final[0])):
        if final[i][j] == 0:
            final[i][j] = 255
        else:
            final[i][j] = 0


plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('original image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(borders, cmap='gray')
plt.title('borders image'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(final, cmap='gray')
plt.title('Filled Image'), plt.xticks([]), plt.yticks([])

plt.show()