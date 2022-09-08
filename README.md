<img alt="face-recognition-plugin" src="https://user-images.githubusercontent.com/82228271/189012182-7cd4d760-90d1-4f78-8003-1e01538c3321.png">

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

