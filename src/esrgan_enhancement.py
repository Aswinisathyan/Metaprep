from realesrgan import RealESRGAN
from PIL import Image
import torch
import os

def enhance_images(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth')

    for file in os.listdir(input_folder):
        try:
            img = Image.open(os.path.join(input_folder, file)).convert("RGB")
            sr_image = model.predict(img)
            sr_image.save(os.path.join(output_folder, file))
        except:
            continue
