<img alt="face-recognition-plugin" src="https://user-images.githubusercontent.com/82228271/190843751-a73de915-f3dc-485f-a63b-8a89a48b6882.png">

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

