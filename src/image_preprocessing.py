import os
from PIL import Image

def preprocess_images(input_folder, output_folder, size=(256, 256)):

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):

        path = os.path.join(input_folder, file)

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(size)
            img.save(os.path.join(output_folder, file))
        except:
            continue
