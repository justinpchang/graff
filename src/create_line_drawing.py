import cv2
import numpy as np
from matplotlib import pyplot as plt


def invert(img):
    return cv2.bitwise_not(img)


def canny(img):
    edges = cv2.Canny(img, 300, 100)
    return edges


def erode(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion = cv2.erode(img, kernel, iterations=3)
    return erosion


def dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(img, kernel, iterations=3)
    return dilation


def open_(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def contour(img):
    result_fill = np.ones(img.shape, np.uint8) * 255
    result_borders = np.zeros(img.shape, np.uint8)
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0][:-1]
    cv2.drawContours(result_fill, contours, -1, 0, -1)
    cv2.drawContours(result_borders, contours, -1, 255, 1)
    edges = result_fill ^ result_borders
    return edges


img = cv2.imread('./assets/test_square.jpg', 0)
img = invert(img)
result = dilate(img)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.title('Result')
plt.xticks([])
plt.yticks([])

plt.show()