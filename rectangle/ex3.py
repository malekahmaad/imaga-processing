import cv2
import numpy as np

def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int( np.ceil(np.sqrt(width * width + height * height)) )  # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
        # Calculate rho. diag_len is added for a positive index
            rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos

def compute_point(theta1, theta2, rho1, rho2):
    a = rho1 / np.sin(theta1)
    b = rho2 / np.sin(theta2)
    c = (-1 * np.cos(theta1) / np.sin(theta1))
    d = (np.cos(theta2) / np.sin(theta2))
    e = c + d
    x = (b - a) / e
    y = c * x + a
    return x, y

kernel = np.ones((5,5),np.uint8)

image = cv2.imread("class_ex6_ner.png", cv2.IMREAD_GRAYSCALE)
ret, th = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)
dilation = cv2.dilate(th, kernel, iterations=2)
edges = cv2.Canny(dilation, 50, 150)

accumulator, thetas, rhos = hough_line(edges)

max_idx = np.argmax(accumulator)
rho_idx, theta_idx = np.unravel_index(max_idx, accumulator.shape)
rho1 = rhos[rho_idx]
theta1 = thetas[theta_idx]

column = accumulator[:, theta_idx]
best_two = np.argsort(column)[-2:][::-1]
theta2 = theta1
rho2 = rhos[best_two[1]]

if theta_idx > 90:
    second_theta = theta_idx - 90

else:
    second_theta = theta_idx + 90

column2 = accumulator[:, second_theta]
best_two2 = np.argsort(column2)[-2:][::-1]
theta3 = thetas[second_theta]
rho3 = rhos[best_two2[0]]
theta4 = thetas[second_theta]
rho4 = rhos[best_two2[1]]

resultx, resulty = compute_point(theta1, theta3, rho1, rho3)
x1 = int(resultx)
y1 = int(resulty)

resultx, resulty = compute_point(theta1, theta4, rho1, rho4)
x2 = int(resultx)
y2 = int(resulty)

resultx, resulty = compute_point(theta2, theta3, rho2, rho3)
x3 = int(resultx)
y3 = int(resulty)

resultx, resulty = compute_point(theta2, theta4, rho2, rho4)
x4 = int(resultx)
y4 = int(resulty)

red_image = cv2.imread('class_ex6_ner.png')
red_image[y1][x1] = [0, 0, 255]
red_image[y2][x2] = [0, 0, 255]
red_image[y3][x3] = [0, 0, 255]
red_image[y4][x4] = [0, 0, 255]
cv2.circle(red_image, (x1, y1), 2, [0, 0, 255], -1)
cv2.circle(red_image, (x2, y2), 2, [0, 0, 255], -1)
cv2.circle(red_image, (x3, y3), 2, [0, 0, 255], -1)
cv2.circle(red_image, (x4, y4), 2, [0, 0, 255], -1)

cv2.imshow("image", edges)
cv2.imshow('image', red_image)
cv2.waitKey(0)
cv2.destroyAllWindows()