# CleanOCR

## Installation
```
pip install cleanocr
```

## Documentation
```
import cv2
from cleanocr import denoise_ocr

image = cv2.imread('test.png')
result = denoise_ocr(image)
cv2.imwrite('result.png', result)
```

## How it works
![example](example/cleanocr.png)

