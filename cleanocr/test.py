import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .models import Generator

MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE = 256


class Transform():
    def __init__(self, resize=RESIZE, mean=MEAN, std=STD):
        self.data_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img: Image.Image):
        return self.data_transform(img)


def de_norm(img):
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach().numpy()
    img_ = np.transpose(img_, (1, 2, 0))
    return img_


def load_model(name, device):
    G = Generator()
    G.load_state_dict(torch.load(f"./cleanocr/checkpoints/G{name}.pth", map_location={"cuda:0": "cpu"}))
    G.eval()
    return G.to(device)


def denoise_ocr(image):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transformer = Transform()
    G = load_model('191', device)

    with torch.no_grad():
        img = transformer(Image.fromarray(image))
        img = img.unsqueeze(0).to(device)
        res_img = G(img)
        output_img = (255 * de_norm(res_img[0].cpu())).astype(np.uint8)
        output_img = cv2.resize(output_img, (400, 400))

    return output_img
