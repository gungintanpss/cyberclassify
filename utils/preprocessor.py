import re
import os
import streamlit as st
from langdetect import detect, LangDetectException, DetectorFactory

# Reproducibility langdetect
DetectorFactory.seed = 42


def cleaning(text: str) -> str:
    # Hapus emoji
    text = re.sub(
        r'[\U0001F000-\U0001FFFF'
        r'\U00002700-\U000027BF'
        r'\U00002600-\U000026FF'
        r'\U00002500-\U0000257F'
        r'\U00002190-\U000021FF'
        r'\U00002000-\U0000206F]+',
        '', text
    )

    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com)\S*', '', text, flags=re.IGNORECASE)

    # Hapus mention (@user)
    text = re.sub(r'@\w+', '', text)

    # Hapus simbol hashtag (#), pertahankan kata hashtag-nya
    text = re.sub(r'#(\w+)', r'\1', text)

    # Hapus karakter non-ASCII
    text = re.sub(r'[^\x00-\x7f]', '', text)

    # Hapus HTML entities (&amp; &lt; dll)
    text = re.sub(r'&\w+;', '', text)

    # Normalisasi karakter berulang
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # "looove" → "loove"

    # Hapus extra whitespace & trim
    text = re.sub(r'\s+', ' ', text).strip()

    # Hapus teks non-English
    if text:
        try:
            if detect(text) != 'en':
                return ""
        except LangDetectException:
            return ""

    # Hapus tweet terlalu pendek
    if len(text.split()) < 3:
        return ""

    return text


def preprocess(text: str) -> str:
    """
    Pipeline identik dengan training di Colab:
    hanya cleaning saja.
    """
    text = cleaning(text)
    if not text:
        return ""
    return text