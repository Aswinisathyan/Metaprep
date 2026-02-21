import os
import numpy as np
import cv2
from PIL import Image


# ======================================================
# IMAGE QUALITY METRICS (Same as before)
# ======================================================

def estimate_blur(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimate_noise(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def compute_image_quality(image):
    blur = estimate_blur(image)
    noise = estimate_noise(image)
    width, height = image.size

    blur_score = min(blur / 150, 1.0)
    noise_score = 1 - min(noise / 100, 1.0)
    resolution_score = min((width * height) / (512 * 512), 1.0)

    IQ = 0.4 * blur_score + 0.3 * noise_score + 0.3 * resolution_score

    return round(max(min(IQ, 1.0), 0.0), 3)


# ======================================================
# OPENCV ENHANCEMENT AGENT
# ======================================================

def enhance_images(input_folder, output_folder):

    if not os.path.exists(input_folder):
        return {
            "status": "failed",
            "reason": "Image folder not found"
        }

    os.makedirs(output_folder, exist_ok=True)

    before_scores = []
    after_scores = []

    valid_ext = (".jpg", ".jpeg", ".png")

    for file in os.listdir(input_folder):

        if not file.lower().endswith(valid_ext):
            continue

        path = os.path.join(input_folder, file)

        try:
            img_pil = Image.open(path).convert("RGB")
            img = np.array(img_pil)

            # -------------------------
            # IQ BEFORE
            # -------------------------
            iq_before = compute_image_quality(img_pil)
            before_scores.append(iq_before)

            # -------------------------
            # 1. DENOISING
            # -------------------------
            denoised = cv2.fastNlMeansDenoisingColored(
                img, None, 10, 10, 7, 21
            )

            # -------------------------
            # 2. CLAHE (Contrast)
            # -------------------------
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # -------------------------
            # 3. SHARPEN
            # -------------------------
            kernel = np.array([[0,-1,0],
                               [-1,5,-1],
                               [0,-1,0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            enhanced_pil = Image.fromarray(enhanced)

            # Save
            save_path = os.path.join(output_folder, file)
            enhanced_pil.save(save_path)

            # -------------------------
            # IQ AFTER
            # -------------------------
            iq_after = compute_image_quality(enhanced_pil)
            after_scores.append(iq_after)

        except:
            continue

    if not before_scores:
        return {
            "status": "no_images",
            "IQ_before": 1.0,
            "IQ_after": 1.0,
            "improvement": 0.0,
            "enhanced_folder": input_folder
        }

    avg_before = round(np.mean(before_scores), 3)
    avg_after = round(np.mean(after_scores), 3)
    improvement = round(avg_after - avg_before, 3)

    return {
        "status": "success",
        "IQ_before": avg_before,
        "IQ_after": avg_after,
        "improvement": improvement,
        "enhanced_folder": output_folder
    }