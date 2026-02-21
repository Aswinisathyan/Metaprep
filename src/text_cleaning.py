import pandas as pd
import re


# =====================================================
# BASIC NORMALIZATION UTILITIES
# =====================================================

def remove_repeated_punctuation(text: str) -> str:
    # Convert !!!!! -> !
    return re.sub(r'([!?.,])\1+', r'\1', text)


def remove_consecutive_word_repetition(text: str) -> str:
    # good good good -> good
    tokens = text.split()
    cleaned_tokens = []

    for token in tokens:
        if not cleaned_tokens or token.lower() != cleaned_tokens[-1].lower():
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def remove_extraneous_characters(text: str) -> str:
    # Remove non-standard characters (keep letters, numbers, punctuation)
    return re.sub(r'[^\w\s.,!?]', '', text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def normalize_case(text: str) -> str:
    return text.lower()


# =====================================================
# ADVANCED TEXT CLEANING PIPELINE
# =====================================================

def clean_text_entry(text: str) -> str:

    if not isinstance(text, str):
        return text

    # Step 1: Lowercase normalization
    text = normalize_case(text)

    # Step 2: Remove repeated punctuation
    text = remove_repeated_punctuation(text)

    # Step 3: Remove repeated consecutive words
    text = remove_consecutive_word_repetition(text)

    # Step 4: Remove extraneous characters
    text = remove_extraneous_characters(text)

    # Step 5: Normalize whitespace
    text = normalize_whitespace(text)

    return text


# =====================================================
# APPLY CLEANING TO DATAFRAME
# =====================================================

def clean_text_dataset(df: pd.DataFrame) -> pd.DataFrame:

    text_cols = df.select_dtypes(include="object")

    if text_cols.empty:
        return df

    for col in text_cols.columns:
        df[col] = df[col].astype(str).apply(clean_text_entry)

    return df
