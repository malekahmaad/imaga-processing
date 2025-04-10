import numpy as np
import cv2


def question1(picture_name, new_picture_name):
    image = cv2.imread(picture_name, cv2.IMREAD_GRAYSCALE)
    # print(image)
    clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(18, 1))
    cl1 = clahe.apply(image)
    blur = cv2.GaussianBlur(cl1, (21, 21), 0)
    ret, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    res = np.hstack((image, th))
    cv2.imwrite(new_picture_name, res)


question1("A.jpg", "newA.jpg")
question1("B.jpg", "newB.jpg")
question1("C.jpg", "newC.jpg")
