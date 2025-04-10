import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def detect_and_draw_contours(image_path: str):
    # img = cv2.imread(image_path)
    # if img is None:
    #     raise ValueError("Wrong image path")

    # # Convert to HSV and extract S and V channels
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h_value, s_value, v_value = cv2.split(hsv)
    # # h_value[:] = 150
    # # v_value[:] = 255
    # # s_value[:] = 100

    # # _, s_value = cv2.threshold(s_value, 20, 255, cv2.THRESH_BINARY)
    # # s_value += 50
    # # h_value += 50

    # combined_weighted = cv2.addWeighted(s_value, 0.8, h_value, 0.2, 0)
    # kernel = np.ones((7,7),np.uint8)
    # erode = cv2.erode(combined_weighted, kernel, iterations=3)

    # edges = combined_weighted - erode
    # # _, th = cv2.threshold(edges, 250, 255, cv2.THRESH_BINARY)
    # # edges = cv2.Canny(combined_weighted, 50, 150)
    # combined_weighted2 = cv2.addWeighted(combined_weighted, 0.6, v_value, 0.4, 0)
    # hsv_modified = cv2.merge([h_value, s_value, v_value])
    # # # Convert back to RGB
    # # img_hue_white = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2RGB)
    # # hsvNew = cv2.cvtColor(img_hue_white, cv2.COLOR_RGB2HSV)
    # # h_valueNew, s_valueNew, v_valueNew = cv2.split(hsvNew)
    # # combined_weightedNew = cv2.addWeighted(s_valueNew, 0.8, v_valueNew, 0.2, 0)
    # # img_hue_white = cv2.cvtColor(img_hue_white, cv2.COLOR_RGB2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(8, 8))
    # cl1 = clahe.apply(combined_weighted)
    # _, th = cv2.threshold(cl1, 200, 255, cv2.THRESH_BINARY_INV)
    # edges = cv2.Canny(edges, 50, 150)

    # # Step 8: Display Results
    # plt.figure(figsize=(15, 6))
    # plt.subplot(161), plt.imshow(img[:, :, ::-1])
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(162), plt.imshow(h_value, cmap='gray')
    # plt.title('h'), plt.xticks([]), plt.yticks([])
    # plt.subplot(163), plt.imshow(s_value, cmap='gray')
    # plt.title('s'), plt.xticks([]), plt.yticks([])
    # plt.subplot(164), plt.imshow(combined_weighted, cmap='gray')
    # plt.title('v'), plt.xticks([]), plt.yticks([])
    # plt.subplot(165), plt.imshow(th, cmap="gray")
    # plt.title('s+h'), plt.xticks([]), plt.yticks([])
    # plt.subplot(166), plt.imshow(edges, cmap="gray")
    # plt.title('s+v'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # # return contours
    
    # second way
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error loading image.")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_value, s_value, v_value = cv2.split(hsv)
    combined_weighted = cv2.addWeighted(s_value, 0.8, h_value, 0.2, 0)
    # Step 1: Choose the best channel for segmentation
    # Based on previous experiments, S-channel (saturation) often works best
    processed = combined_weighted.copy()

    # Step 2: Apply CLAHE to enhance weak contrasts
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(25,4))
    processed = clahe.apply(processed)

    # Step 3: Apply thresholding to separate circles from the background
    _, binary_thresh = cv2.threshold(processed, 150, 255, cv2.THRESH_BINARY)

    # Step 4: Morphological operations to improve shape continuity
    kernel = np.ones((15, 15), np.uint8)
    binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_OPEN, kernel)
    # kernel2 = np.ones((5, 5), np.uint8)
    # binary_thresh3 = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel2, iterations=100)

    # binary_thresh = binary_thresh - binary_thresh3

    # Step 5: Find contours
    contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Filter contours based on size and shape (circularity)
    detected_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if 0.22 < circularity:  # Filtering circular objects
            print(circularity)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_perimeter = 2 * np.pi * radius
            if 50 < radius < 130 and 0.6 < circle_perimeter / perimeter < 1.1:
                detected_circles.append((int(x), int(y), int(radius)))

    # Step 7: Draw detected circles
    output_image = img.copy()
    for (x, y, r) in detected_circles:
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)  # Green circle

    # Step 8: Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(binary_thresh, cmap='gray')
    plt.title('Processed Binary Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles (Contours)'), plt.xticks([]), plt.yticks([])
    plt.show()

    # return detected_circles

    #third way
    # img = cv2.imread(image_path)
    # if img is None:
    #     raise ValueError("Error loading image.")

    # # Convert to HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h_value, s_value, v_value = cv2.split(hsv)
    
    # # Step 1: Select the Best Channel (S or V)
    # combined_weighted = cv2.addWeighted(s_value, 0.9, h_value, 0.1, 0)
    # # combined_weighted = cv2.addWeighted(combined_weighted, 0.7, s_value, 0.5, 0)

    # # Step 2: Apply CLAHE for better contrast
    # clahe = cv2.createCLAHE(clipLimit=12, tileGridSize=(10, 10))
    # processed = clahe.apply(combined_weighted)

    # _, binary_thresh = cv2.threshold(processed, 150, 255, cv2.THRESH_BINARY)

    # # Step 3: Apply Gaussian Blur to Reduce Noise
    # # blurred = cv2.GaussianBlur(processed, (9, 9), 2)

    # # Step 4: Apply Canny Edge Detection
    # edges = cv2.Canny(binary_thresh, 50, 150)


    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour_image = img.copy()
    # # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    # output_image = img.copy()
    # min_radius_threshold = 50
    # min_circularity_threshold = 0.1  # Ideal circles are close to 1

    # # Loop through each contour individually
    # print(len(contours))
    # for contour in contours:
    #     # Create a new black image for each contour
    #     # if len(contour) <= 100:
    #     #     continue
    #     if len(contour) > 100:
    #         black_image = np.zeros_like(binary_thresh)
    #         cv2.drawContours(black_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    #         # plt.figure(figsize=(12, 6))
    #         # # plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         # # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #         # plt.subplot(132), plt.imshow(black_image, cmap='gray')
    #         # plt.title('Processed Binary Image'), plt.xticks([]), plt.yticks([])
    #         # # plt.subplot(133), plt.imshow(output_image)
    #         # # plt.title('Contours'), plt.xticks([]), plt.yticks([])
    #         # # Draw the current contour in white
    #         # plt.show()

    #         # Apply Hough Circle Transform
    #         circles = cv2.HoughCircles(
    #             black_image, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
    #             param1=50, param2=30, minRadius=10, maxRadius=100
    #         )

    #         # Compute contour circularity
    #         area = cv2.contourArea(contour)
    #         perimeter = cv2.arcLength(contour, True)

    #         if perimeter == 0 or area == 0:
    #             continue  # Skip invalid contours

    #         circularity = (4 * np.pi * area) / (perimeter ** 2)  # Circularity formula

    #         # Check if we detected a circle
    #         if circles is not None:
    #             print("entered")
    #             circles = np.uint16(np.around(circles))
    #             for circle in circles[0, :]:
    #                 x, y, radius = circle
    #                 circle_perimeter = 2 * np.pi * radius  # Perimeter of Hough circle

    #                 # Validate if contour is circular and similar to the detected Hough circle
    #                 if (circularity > min_circularity_threshold and radius > min_radius_threshold
    #                         and abs(perimeter - circle_perimeter) < 15):  # Allow slight perimeter difference

    #                     # Draw the detected circle and valid contour on the original image
    #                     cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)  # Green Circle
    #                     cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)  # Mark Center
    #                     cv2.drawContours(output_image, [contour], -1, (255, 0, 0), 2)  # Blue Contour


    # Step 5: Apply Hough Circle Transform
    # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=350,
    #                            param1=50, param2=25, minRadius=60, maxRadius=80)

    # detected_circles = []
    # output_image = img.copy()

    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for circle in circles[0, :]:
    #         x, y, r = circle
    #         detected_circles.append((x, y, r))
    #         # cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)  # Draw circle
    #         # cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)  # Draw center

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)

    #     if perimeter == 0:
    #         # print(contour)
    #         continue  # Avoid division by zero

    #     # print("got here")
    #     # Compute circularity
    #     circularity = (4 * np.pi * area) / (perimeter ** 2)

    #     # Fit a minimum enclosing circle
    #     (x, y), radius = cv2.minEnclosingCircle(contour)
    #     radius = int(radius)
    #     print(circularity)
    #     print(radius)
    #     print()
    #     # Check circularity and radius conditions
    #     if radius > 30:
    #         cv2.circle(output_image, (int(x), int(y)), radius, (0, 255, 0), 2)  # Draw circle
    #         cv2.circle(output_image, (int(x), int(y)), 2, (0, 0, 255), 3)  # Mark center

    # Step 6: Display Results
    # plt.figure(figsize=(12, 6))
    # plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(edges, cmap='gray')
    # plt.title('Processed Binary Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(output_image)
    # plt.title('Contours'), plt.xticks([]), plt.yticks([])
    # # plt.subplot(134), plt.imshow(output_image)
    # # plt.title('Contours'), plt.xticks([]), plt.yticks([])
    # plt.show()


def main():
    # print(detect_and_draw_contours("random_004.jpg"))
    # print(detect_and_draw_contours("random_005.jpg"))
    # # print(detect_colored_circles("seq_000.jpg"))
    # print(detect_and_draw_contours("random_003.jpg"))
    # print(detect_and_draw_contours("random_008.jpg"))
    # print(detect_and_draw_contours("seq_000.jpg"))
    # print(detect_and_draw_contours("seq_001.jpg"))
    # print(detect_and_draw_contours("seq_002.jpg"))
    # print(detect_and_draw_contours("seq_003.jpg"))
    # print(detect_and_draw_contours("seq_004.jpg"))
    # print(detect_and_draw_contours("seq_005.jpg"))
    # print(detect_and_draw_contours("seq_006.jpg"))
    # print(detect_and_draw_contours("seq_007.jpg"))
    # print(detect_and_draw_contours("seq_008.jpg"))
    # print(detect_and_draw_contours("seq_009.jpg"))
    # result = track_circles_over_time(["img1.jpg", "img2.jpg", "img3.jpg"]) 
    # print(result["table"])
    folder_path = "random"
    images = [f for f in os.listdir(folder_path)]
    for image in images:
        detect_and_draw_contours(f"{folder_path}/{image}")

if __name__ == "__main__":
    main()
            