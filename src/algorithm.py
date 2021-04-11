import cv2
import numpy as np
import glob
import os.path
from matplotlib import pyplot as plt
from skimage import metrics


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


def prepare(img):
    return center(clean(binarize(img)))


def hausdorff(img1, img2):
    return metrics.hausdorff_distance(img1, img2)


def minimize(img1, img1_filename):
    ds = {}
    for filename in glob.glob('./assets/*.jpg'):
        if filename == img1_filename:
            continue
        img2 = cv2.imread(filename, 0)
        d = hausdorff(prepare(img1), prepare(img2))
        ds[filename] = d
    min_d = min(ds.values())
    return (min_d, [filename for filename in ds if ds[filename] == min_d])


input_filename = './assets/test_star.jpg'
img1 = cv2.imread(input_filename, 0)
min_d, matches = minimize(img1, input_filename)

plt.figure()

plt.subplot(121)
plt.imshow(img1, cmap='gray', interpolation='none')
plt.title('Input')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(prepare(img1), cmap='gray', interpolation='none')
for match in matches:
    img = prepare(cv2.imread(match, 0))
    plt.imshow(img, cmap='jet', interpolation='none', alpha=0.5)
plt.title('Matches: ' + ','.join(matches))
plt.xticks([])
plt.yticks([])

plt.suptitle('d_H(img1, img2) = ' + str(min_d), y=0.1)

plt.show()