"""
name: Malek Ahmad
ID: 324921345
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def check_existense(circles, x, y):
    for circle in circles:
        if np.abs(circle["x"] - x) < 5 and np.abs(circle["y"] - y) < 5:
            return True

    return False

def get_circle_color_in_yellow_image(img, x, y):
    position = img[int(y)][int(x)]
    blue_green = np.abs(position[0].astype(np.int32) - position[1].astype(np.int32))
    blue_red = np.abs(position[0].astype(np.int32) - position[2].astype(np.int32))
    green_red = np.abs(position[1].astype(np.int32) - position[2].astype(np.int32))
    # print(position, x, y, blue_green, blue_red, green_red)
    if blue_green > 15 and blue_red > 15 and green_red >= 6:
        return "red"
        
    elif blue_green > 15 and blue_red > 15 and green_red >= 0:
        return "green"
    
    else:
        return "blue"


def get_circle_color(img, x, y):
    position = img[int(y)][int(x)]
    blue_green = np.abs(position[0].astype(np.int32) - position[1].astype(np.int32))
    blue_red = np.abs(position[0].astype(np.int32) - position[2].astype(np.int32))
    green_red = np.abs(position[1].astype(np.int32) - position[2].astype(np.int32))
    # print(position, x, y, blue_green, blue_red, green_red)
    if blue_green > 19 and blue_red > 20:
        return "blue"
        
    elif blue_green > 10 and green_red > 10 and blue_red < 20:
        return "green"
    
    elif blue_red > 20 and green_red > 20:
        return "red"
    
    elif blue_red > 8:
        return "blue"


def detect_colored_circles(image_path: str) -> list[dict]:
    circles = []
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("wrong path, couldnt find image")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_value, s_value, v_value = cv2.split(hsv)

    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
    yellow_pixels = cv2.countNonZero(yellow_mask)
    total_pixels = img.shape[0] * img.shape[1]
    yellow_ratio = yellow_pixels / total_pixels

    if yellow_ratio > 0.3:
        # print("yellow background")
        combined_weighted = cv2.addWeighted(s_value, 1.5, h_value, 0.2, 0)

        processed = combined_weighted.copy()
        
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(20,5))
        processed = clahe.apply(processed)

        _, binary_thresh = cv2.threshold(processed, 60, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((6, 6), np.uint8)
        binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_OPEN, kernel, iterations=7)
        binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel, iterations=15)

        edges = cv2.Canny(binary_thresh, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # contour_image = img.copy()
        # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        # print(len(contours))
        detected_circles = []
        for cnt in contours:
            # print("entered")
            circle = dict()
            # area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            # circularity = 4 * np.pi * (area / (perimeter ** 2))
            # if 0.22 < circularity:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            exist = check_existense(circles, x, y)
            if exist == True:
                continue
            
            # circle_perimeter = 2 * np.pi * radius
            # if 50 < radius < 130 and 0.6 < circle_perimeter / perimeter < 1.1:
            circle["x"] = x
            circle["y"] = y
            circle["radius"] = radius
            detected_circles.append((int(x), int(y), int(radius)))

            circle["color"] = get_circle_color_in_yellow_image(img, x, y)
            if circle["color"] != None:
                circles.append(circle)            

        # output_image = img.copy()
        # for (x, y, r) in detected_circles:
        #     cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)

        # plt.figure(figsize=(12, 6))
        # plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(132), plt.imshow(edges, cmap='gray')
        # plt.title('Processed Binary Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(133), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        # plt.title('Detected Circles (Contours)'), plt.xticks([]), plt.yticks([])
        # plt.show()

        return circles
    else:
        combined_weighted = cv2.addWeighted(s_value, 0.8, h_value, 0.2, 0)
        processed = combined_weighted.copy()

        clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(25,4))
        processed = clahe.apply(processed)

        _, binary_thresh = cv2.threshold(processed, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)
        binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_OPEN, kernel)
    
        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_circles = []
        for cnt in contours:
            circle = dict()
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if 0.22 < circularity:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circle_perimeter = 2 * np.pi * radius
                if 50 < radius < 130 and 0.6 < circle_perimeter / perimeter < 1.1:
                    circle["x"] = x
                    circle["y"] = y
                    circle["radius"] = radius
                    detected_circles.append((int(x), int(y), int(radius)))

                    circle["color"] = get_circle_color(img, x, y)
                    if circle["color"] != None:
                        circles.append(circle)            

        # output_image = img.copy()
        # for (x, y, r) in detected_circles:
        #     cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)

        # plt.figure(figsize=(12, 6))
        # plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(132), plt.imshow(binary_thresh, cmap='gray')
        # plt.title('Processed Binary Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(133), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        # plt.title('Detected Circles (Contours)'), plt.xticks([]), plt.yticks([])
        # plt.show()

        return circles

def track_circles_over_time(image_paths: list[str]) -> dict:
    image_count = 1
    circles = list()
    result = dict()
    images_list = list()
    for image in image_paths:
        circles_results = detect_colored_circles(image)
        for circle in circles_results:
            circle_index = len(circles) + 1
            exist = False 
            for i in range(len(circles)):
                if circle["color"] == circles[i]["color"]:
                    if np.abs(circle["x"] - circles[i]["x"]) <= 100 and np.abs(circle["y"] - circles[i]["y"]) <= 100:
                        circle_index = i + 1
                        exist = True
                        break

            if exist is False:
                circles.append(circle)

            circle_data = dict()
            circle_data["image"] = image_count
            circle_data["circle_id"] = circle_index
            circle_data["x"] = circle["x"]
            circle_data["y"] = circle["y"]
            circle_data["radius"] = circle["radius"]
            circle_data["color"] = circle["color"]
            images_list.append(circle_data)

        image_count += 1

    result["table"] = images_list

    return result


def main():
    # folder_path = "random"
    # images = [f for f in os.listdir(folder_path)]
    # for image in images:
    #     print(detect_colored_circles(f"{folder_path}/{image}"))
    # folder_path = "seq_4"
    # images = [f"{folder_path}/{f}" for f in os.listdir(folder_path)]
    # result = track_circles_over_time(images) 
    # print(result["table"]) 
    result = track_circles_over_time(["seq_1/seq_000.jpg", "seq_1/seq_003.jpg"]) 
    print(result["table"]) 


if __name__ == "__main__":
    main()
            