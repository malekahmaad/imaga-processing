"""
malek ahmad
324921345
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos

def compute_point(theta1, theta2, rho1, rho2):
    if np.sin(theta1) == 0:
        return rho1/np.cos(theta1), rho2/np.sin(theta2)
    
    if np.sin(theta2) == 0:
        return rho2/np.cos(theta2), rho1/np.sin(theta1)

    a = rho1 / np.sin(theta1)
    b = rho2 / np.sin(theta2)
    c = (-1 * np.cos(theta1) / np.sin(theta1))
    d = (np.cos(theta2) / np.sin(theta2))
    e = c + d
    x = (b - a) / e
    y = c * x + a
    return x, y

def calculate_area(img):
    image = cv2.imread(img,  cv2.IMREAD_GRAYSCALE)
    red_image = cv2.imread('baffalo.png')
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    ret, th = cv2.threshold(laplacian, 35, 255, cv2.THRESH_BINARY)
    visited = np.zeros_like(th, dtype=bool)

    objects = []

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if th[row, col] > 0 and visited[row, col] == False:
                object_list = []
                object_mask = np.zeros_like(image, dtype=np.uint8)
                stack = [(row, col)]
                while stack:
                    cur_row, cur_col = stack.pop()
                    if th[cur_row, cur_col] > 0 and visited[cur_row, cur_col] == False:
                        visited[cur_row, cur_col] = True
                        object_mask[cur_row, cur_col] = 255
                        object_list.append((cur_row, cur_col))
                        if 0 <= cur_col - 1:
                            stack.append((cur_row, cur_col - 1))
                        
                        if cur_col + 1 < image.shape[1]:
                            stack.append((cur_row, cur_col + 1))
        
                        if 0 <= cur_row - 1:
                            stack.append((cur_row - 1, cur_col))
                        
                        if cur_row + 1 < image.shape[0]:
                            stack.append((cur_row + 1, cur_col))

                if len(object_list) > 30:
                    objects.append(object_mask)
                
    for i in range(len(objects)):
        accumulator, thetas, rhos = hough_line(objects[i])

        max_idx = np.argmax(accumulator)
        rho_idx, theta_idx = np.unravel_index(max_idx, accumulator.shape)
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        
        column = accumulator[:, theta_idx]
        best_two = np.argsort(column)[-2:][::-1]
        theta2 = theta
        rho2 = rhos[best_two[1]]

        if theta_idx > 90:
            second_theta = theta_idx - 90

        else:   
            second_theta = theta_idx + 90
            if second_theta == 180:
                second_theta = 179

        column2 = accumulator[:, second_theta]
        best_two2 = np.argsort(column2)[-2:][::-1]
        theta3 = thetas[second_theta]
        rho3 = rhos[best_two2[0]]
        theta4 = thetas[second_theta]
        rho4 = rhos[best_two2[1]]
   
        resultx, resulty = compute_point(theta, theta3, rho, rho3)
        x1 = int(resultx)
        y1 = int(resulty)

        resultx, resulty = compute_point(theta, theta4, rho, rho4)
        x2 = int(resultx)
        y2 = int(resulty)

        resultx, resulty = compute_point(theta2, theta3, rho2, rho3)
        x3 = int(resultx)
        y3 = int(resulty)

        resultx, resulty = compute_point(theta2, theta4, rho2, rho4)
        x4 = int(resultx)
        y4 = int(resulty)

        points = set()
        points.add((x1,y1))
        points.add((x2,y2))
        points.add((x3,y3))
        points.add((x4,y4))
         
        if len(points) == 4:
            x1_max = max(np.abs(x2-x1), np.abs(x3-x1), np.abs(x4-x1))
            x2_max = max(np.abs(x2-x1), np.abs(x3-x2), np.abs(x4-x2))
            x3_max = max(np.abs(x2-x3), np.abs(x3-x1), np.abs(x4-x3))
            x4_max = max(np.abs(x2-x4), np.abs(x3-x4), np.abs(x4-x1))
            width = max(x1_max, x2_max, x3_max, x4_max)
            y1_max = max(np.abs(y2-y1), np.abs(y3-y1), np.abs(y4-y1))
            y2_max = max(np.abs(y2-y1), np.abs(y3-y2), np.abs(y4-y2))
            y3_max = max(np.abs(y2-y3), np.abs(y3-y1), np.abs(y4-y3))
            y4_max = max(np.abs(y2-y4), np.abs(y3-y4), np.abs(y4-y1))
            height = max(y1_max, y2_max, y3_max, y4_max)
            if 400 < width * height < 700:
                print(f"area of the L letter is equal to {width * height}")
                cv2.circle(red_image, (x1, y1), 2, [255, 0, 0], -1)
                cv2.circle(red_image, (x2, y2), 2, [255, 0, 0], -1)
                cv2.circle(red_image, (x3, y3), 2, [255, 0, 0], -1)
                cv2.circle(red_image, (x4, y4), 2, [255, 0, 0], -1)

    plt.figure()
    plt.imshow(red_image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


calculate_area("baffalo.png")