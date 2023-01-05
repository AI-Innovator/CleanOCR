import os
import sys
sys.path.append('.')
import glob
import cv2
import numpy as np
from cleanocr import denoise_ocr, thinning, binarization


def test():
    image = cv2.imread('unit_test/test.png')
    result = denoise_ocr(image)
    cv2.imwrite('unit_test/result.png', result)


def test_org():
    image = cv2.imread('unit_test/test1.png')
    img_h, img_w, img_c = image.shape

    dst_img = np.zeros(image.shape)
    nh_size = 400
    nw_size = 400

    for i in range(0, img_h, nh_size):
        for j in range(0, img_w, nw_size):
            x1 = j
            x2 = j + nw_size
            y1 = i
            y2 = i + nh_size

            if x2 >= img_w:
                x1 = img_w - nw_size
                x2 = img_w

            if y2 >= img_h:
                y1 = img_h - nh_size
                y2 = img_h

            crop_img = image[y1:y2, x1:x2]
            result = denoise_ocr(crop_img)

            dst_img[y1:y2, x1:x2] = result

    cv2.imwrite('unit_test/result.png', dst_img)
    sys.exit(1)


def test_dir():
    image_list = glob.glob('unit_test/train_cleaned/*.png')
    for path in image_list:
        filename = os.path.basename(path)
        image = cv2.imread(path)
        result = denoise_ocr(image)
        dst_path = os.path.join('unit_test/train_result', filename)
        cv2.imwrite(dst_path, result)


def test_thinning():
    image = cv2.imread('unit_test/result.png')
    result = thinning(image)
    cv2.imwrite('unit_test/thin.png', result)


def test_binarization():
    image = cv2.imread('unit_test/result.png')
    result = binarization(image)
    cv2.imwrite('unit_test/binary.png', result)


if __name__ == '__main__':
    test()
    test_org()
    test_thinning()
    test_binarization()

