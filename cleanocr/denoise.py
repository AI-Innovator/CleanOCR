import os
import cv2
import requests
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from .models import Generator, SRCNN

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


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def download_checkpoint(remote_url, local_path):
    response = requests.get(remote_url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(
        desc=f"Downloading {local_path}..",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(local_path, "wb") as ref:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            ref.write(data)

    progress_bar.close()
    if total_size_in_bytes not in (0, progress_bar.n):
        print("ERROR, something went wrong")


def load_g_model(device):
    url = "https://github.com/AI-Innovator/CleanOCR/releases/download/v0.1.3/G.pth"
    resume_path = os.path.join(os.path.dirname(__file__), 'checkpoints/G.pth')
    if not os.path.exists(resume_path):
        os.makedirs(os.path.dirname(resume_path), exist_ok=True)
        download_checkpoint(remote_url=url, local_path=resume_path)

    G = Generator()
    G.load_state_dict(torch.load(resume_path, map_location={"cuda:0": "cpu"}))
    G.eval()
    return G.to(device)


def load_s_model(device):
    url = "https://github.com/AI-Innovator/CleanOCR/releases/download/v0.1.3/S.pth"
    resume_path = os.path.join(os.path.dirname(__file__), 'checkpoints/S.pth')
    if not os.path.exists(resume_path):
        os.makedirs(os.path.dirname(resume_path), exist_ok=True)
        download_checkpoint(remote_url=url, local_path=resume_path)

    S = SRCNN()
    S.load_state_dict(torch.load(resume_path, map_location={"cuda:0": "cpu"}))
    S.eval()
    return S.to(device)


def denoise_ocr(image):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transformer = Transform()
    G = load_g_model(device)
    S = load_s_model(device)

    h, w, c = image.shape
    with torch.no_grad():
        img = transformer(Image.fromarray(image))
        img = img.unsqueeze(0).to(device)
        res_img = G(img)
        output_img = (255 * de_norm(res_img[0].cpu())).astype(np.uint8)
        output_img = cv2.resize(output_img, (400, 400))

        image_np = np.array(Image.fromarray(output_img).convert('RGB')).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image_np)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)
        preds = S(y).clamp(0.0, 1.0)

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

    output = cv2.resize(output, (h, w))
    return output
