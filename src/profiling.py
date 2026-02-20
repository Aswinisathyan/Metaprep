import pandas as pd
import os
from PIL import Image
import numpy as np
import cv2
import re

# ======================================================
# TEXT PROFILING
# ======================================================

def profile_text_dataset(df: pd.DataFrame):

    profile = {}

    total_cells = df.size
    total_rows = len(df)

    # -----------------------------
    # COMPLETENESS (C)
    # -----------------------------
    missing = df.isnull().sum().sum()
    C = 1 - (missing / total_cells if total_cells else 0)
    profile["C"] = round(max(min(C, 1), 0), 3)

    # -----------------------------
    # CONSISTENCY (S)
    # -----------------------------
    duplicates = df.duplicated().sum()
    S = 1 - (duplicates / total_rows if total_rows else 0)
    profile["S"] = round(max(min(S, 1), 0), 3)

    # -----------------------------
    # TEXT METRICS
    # -----------------------------
    text_cols = df.select_dtypes(include="object")

    if text_cols.empty:
        profile.update({
            "punctuation_score": 1.0,
            "repetition_score": 1.0,
            "extraneous_char_score": 1.0,
            "length_stability_score": 1.0
        })
        return profile

    all_text = text_cols.apply(lambda col: col.astype(str)).stack()

    total_tokens = 0
    repeated_tokens = 0
    total_punct = 0
    repeated_punct = 0
    abnormal_chars = 0
    total_chars = 0
    lengths = []

    for text in all_text:

        # Length tracking
        lengths.append(len(text))

        # Character tracking
        total_chars += len(text)
        abnormal_chars += len(re.findall(r"[^\w\s.,!?]", text))

        # Repeated punctuation detection (!!!!!)
        repeated_punct += len(re.findall(r"[!?.,]{2,}", text))
        total_punct += len(re.findall(r"[!?.,]", text))

        # Token repetition
        tokens = text.lower().split()
        total_tokens += len(tokens)

        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                repeated_tokens += 1

    # -----------------------------
    # Punctuation Score
    # -----------------------------
    if total_punct > 0:
        P = 1 - (repeated_punct / total_punct)
    else:
        P = 1.0

    # -----------------------------
    # Repetition Score
    # -----------------------------
    if total_tokens > 0:
        R = 1 - (repeated_tokens / total_tokens)
    else:
        R = 1.0

    # -----------------------------
    # Extraneous Character Score
    # -----------------------------
    if total_chars > 0:
        E = 1 - (abnormal_chars / total_chars)
    else:
        E = 1.0

    # -----------------------------
    # Length Stability Score
    # -----------------------------
    avg_length = np.mean(lengths) if lengths else 0
    ideal_length = 50  # configurable baseline

    if ideal_length > 0:
        L = 1 - abs(avg_length - ideal_length) / ideal_length
    else:
        L = 1.0

    profile.update({
        "punctuation_score": round(max(min(P, 1), 0), 3),
        "repetition_score": round(max(min(R, 1), 0), 3),
        "extraneous_char_score": round(max(min(E, 1), 0), 3),
        "length_stability_score": round(max(min(L, 1), 0), 3)
    })

    return profile


# ======================================================
# IMAGE PROFILING
# ======================================================

def estimate_blur(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimate_noise(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def profile_images(image_folder):

    blur_scores = []
    noise_scores = []
    resolutions = []

    for file in os.listdir(image_folder):
        path = os.path.join(image_folder, file)

        try:
            img = Image.open(path)

            blur_scores.append(estimate_blur(img))
            noise_scores.append(estimate_noise(img))
            resolutions.append(img.size)

        except:
            continue

    if not resolutions:
        return {
            "blur_score": 1.0,
            "noise_score": 1.0,
            "resolution_score": 1.0
        }

    avg_blur = np.mean(blur_scores)
    avg_noise = np.mean(noise_scores)
    avg_width = np.mean([r[0] for r in resolutions])
    avg_height = np.mean([r[1] for r in resolutions])

    # Normalize Blur (threshold 100)
    blur_score = min(avg_blur / 100, 1.0)

    # Normalize Noise (lower noise = better)
    noise_score = 1 - min(avg_noise / 100, 1.0)

    # Normalize Resolution (baseline 512x512)
    resolution_score = min((avg_width * avg_height) / (512 * 512), 1.0)

    return {
        "blur_score": round(max(min(blur_score, 1), 0), 3),
        "noise_score": round(max(min(noise_score, 1), 0), 3),
        "resolution_score": round(max(min(resolution_score, 1), 0), 3)
    }
