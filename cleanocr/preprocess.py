
import cv2
import numpy as np
# from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter


def thinning(img):
    """ Thinning and Skeletonization """
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion


def binarization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return bin_img

"""

# Noise Removal
# Reading image from folder where it is stored
img = cv2.imread('bear.png')
# denoising of image saving it into dst image
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
# Plotting of source and destination image


# Skew Correction
input_file = sys.argv[1]
img = im.open(input_file)
# convert to binary
wd, ht = img.size
pix = np.array(img.convert('1').getdata(), np.uint8)
bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
plt.imshow(bin_img, cmap='gray')
plt.savefig('binary.png')
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score
delta = 1
limit = 5
angles = np.arange(-limit, limit+delta, delta)
scores = []
for angle in angles:
    hist, score = find_score(bin_img, angle)
    scores.append(score)
best_score = max(scores)
best_angle = angles[scores.index(best_score)]
print('Best angle: {}'.formate(best_angle))
# correct skew
data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
img.save('skew_corrected.png')

"""
