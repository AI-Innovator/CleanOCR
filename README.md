<img alt="face-recognition-plugin" src="https://user-images.githubusercontent.com/82228271/190843751-a73de915-f3dc-485f-a63b-8a89a48b6882.png">

<details open>
<summary><h2>How it works</h2></summary>

</details>
<details open>
<summary><h2>Installation</h2></summary>
  
```
pip install cleanocr
```
 
</details>
<details open>
<summary><h2>Documentation</h2></summary>

<h3>Denoise OCR</h3>

```
import cv2
from cleanocr import denoise_ocr

image = cv2.imread('test.png')
result = denoise_ocr(image)
cv2.imwrite('result.png', result)
```

<h3>Thinning</h3>

```
import cv2
from cleanocr import thinning

image = cv2.imread('unit_test/result.png')
result = thinning(image)
cv2.imwrite('unit_test/thin.png', result)
```

<h3>Binarization</h3>

```
import cv2
from cleanocr import binarization

image = cv2.imread('unit_test/result.png')
result = binarization(image)
cv2.imwrite('unit_test/unit_test/binary.png.png', result)
```

</details>
<details open>
<summary><h2>Contact</h2></summary>

Please reach out to me for your any projects in Python, Web Development and Computer Vision and NLP fields.
<div align="left">
<a target="_blank" href="https://t.me/jareddean"><img src="https://img.shields.io/badge/telegram-prenes-green.svg?logo=telegram " alt="www.prenes.org"></a>
<a target="_blank" href="https://wa.me/+14422295661"><img src="https://img.shields.io/badge/whatsapp-prenes-green.svg?logo=whatsapp " alt="www.prenes.org"></a>
<a target="_blank" href="https://join.slack.com/t/prenes/shared_invite/zt-1cx925fip-vL4nKJN64XBMbx8vdwHP7Q"><img src="https://img.shields.io/badge/slack-prenes-green.svg?logo=slack " alt="www.prenes.org"></a>
<a target="_blank" href="skype:live:.cid.4b536a6c3cc88a8c?chat"><img src="https://img.shields.io/badge/skype-prenes-green.svg?logo=skype " alt="www.prenes.org"></a>
</div>

</details>
