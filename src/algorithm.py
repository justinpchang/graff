import cv2
import numpy as np
from matplotlib import pyplot as plt


def binarize(img):
    # Generate binary image from grayscale
    (thresh, binary) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Ensure black background
    w, h = binary.shape[:2]
    if cv2.countNonZero(binary) > ((w * h) // 2):
        binary = cv2.bitwise_not(binary)
    return binary


def clean(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img
    # Erosion erases a 1px line drawing
    return opening


def center(img):
    # Get centroid
    moments = cv2.moments(img)
    ctdx = int(moments["m10"] / moments["m00"])
    ctdy = int(moments["m01"] / moments["m00"])
    # Place centroid in center of image mask
    imgh, imgw = img.shape
    dx = ctdx - imgw // 2
    dy = ctdy - imgh // 2
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    centered = cv2.warpAffine(img, M, img.shape)
    return centered


img = cv2.imread('./assets/test_square.jpg', 0)
result = img
result = binarize(result)
result = clean(result)
result = center(result)

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