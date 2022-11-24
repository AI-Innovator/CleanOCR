import os
import sys
sys.path.append('.')
import glob
import cv2
from cleanocr import denoise_ocr, thinning


def test():
    image = cv2.imread('unit_test/test.png')
    result = denoise_ocr(image)
    cv2.imwrite('unit_test/result.png', result)


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


if __name__ == '__main__':
    test()
    test_thinning()

