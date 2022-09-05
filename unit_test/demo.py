import sys
sys.path.append('.')
import cv2
from cleanocr import denoise_ocr


def test():
    image = cv2.imread('unit_test/test.png')
    result = denoise_ocr(image)
    cv2.imwrite('unit_test/result.png', result)


if __name__ == '__main__':
    test()