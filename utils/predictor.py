import os
import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

from utils.preprocessor import preprocess


class CNNInference:

    def __init__(self, weights: dict):
        self.w = weights

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _bn(x: np.ndarray, gamma, beta, mean, var, eps: float = 1e-5) -> np.ndarray:
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    @staticmethod
    def _conv1d(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        batch, length, _ = x.shape
        k                = W.shape[0]
        pad              = (k - 1) // 2
        x_pad            = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        out              = np.zeros((batch, length, W.shape[2]), dtype=np.float32)
        for ki in range(k):
            out += np.matmul(x_pad[:, ki:ki + length, :], W[ki])
        return out + b

    @staticmethod
    def _maxpool1d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        batch, length, ch = x.shape
        out_len   = (length + pool_size - 1) // pool_size
        pad_total = out_len * pool_size - length
        if pad_total > 0:
            pad_left  = pad_total // 2
            pad_right = pad_total - pad_left
            x = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)),
                       constant_values=-np.inf)
        x_r = x[:, :out_len * pool_size, :].reshape(batch, out_len, pool_size, ch)
        return x_r.max(axis=2)

    @staticmethod
    def _global_maxpool(x: np.ndarray) -> np.ndarray:
        return x.max(axis=1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w = self.w

        # Conv Block 1: conv1 → bn1 → relu1 → pool1
        x = self._conv1d(x, w['conv1_W'], w['conv1_b'])
        x = self._bn(x, w['bn1_gamma'], w['bn1_beta'],
                        w['bn1_mean'],  w['bn1_var'])
        x = self._relu(x)
        x = self._maxpool1d(x, pool_size=2)

        # Conv Block 2: conv2 → bn2 → relu2 (tanpa MaxPool)
        x = self._conv1d(x, w['conv2_W'], w['conv2_b'])
        x = self._bn(x, w['bn2_gamma'], w['bn2_beta'],
                        w['bn2_mean'],  w['bn2_var'])
        x = self._relu(x)

        # GlobalMaxPool
        x = self._global_maxpool(x)

        # Dense Head: dense1 → bn3 → relu3
        x = np.dot(x, w['dense1_W']) + w['dense1_b']
        x = self._bn(x, w['bn3_gamma'], w['bn3_beta'],
                        w['bn3_mean'],  w['bn3_var'])
        x = self._relu(x)

        # Output: dense2 → softmax
        x = np.dot(x, w['dense2_W']) + w['dense2_b']
        x = self._softmax(x)

        return x



_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))          # folder utils/
_ELMO_PATH = os.path.join(_BASE_DIR, '..', 'model', 'elmo3')    # → model/elmo3


@st.cache_resource(show_spinner="⏳ Memuat model, harap tunggu...")
def _load_resources():
    elmo_path = os.path.abspath(_ELMO_PATH)

    # Validasi folder ELMo sebelum load
    if not os.path.isfile(os.path.join(elmo_path, 'saved_model.pb')):
        raise FileNotFoundError(
            f"File 'saved_model.pb' tidak ditemukan di '{elmo_path}'.\n"
            "Pastikan folder model/elmo3/ sudah berisi saved_model.pb dan variables/."
        )

    # Load bobot CNN
    weights = np.load(
        os.path.join(_BASE_DIR, '..', 'model', 'cnn_weights.npy'),
        allow_pickle=True
    ).item()

    # Load config
    with open(os.path.join(_BASE_DIR, '..', 'model', 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Load label encoder
    with open(os.path.join(_BASE_DIR, '..', 'model', 'label_encoder.pkl'), 'rb') as f:
        label_enc = pickle.load(f)

    #  Load ELMo dari lokal 
    elmo = hub.load(elmo_path)

    return weights, config, label_enc, elmo


def _get_elmo_embedding(text: str, elmo_module, max_len: int) -> np.ndarray:
    """Ambil ELMo embedding untuk satu teks, lalu pad/truncate ke max_len."""
    elmo_out = elmo_module.signatures['default'](tf.constant([text]))
    emb      = elmo_out['elmo'].numpy()          # (1, token_len, 1024)

    padded = np.zeros((1, max_len, 1024), dtype=np.float32)
    actual = min(emb.shape[1], max_len)
    padded[0, :actual, :] = emb[0, :actual, :]
    return padded


def classify_text(text: str) -> dict:
    weights, config, label_enc, elmo = _load_resources()

    # 1. Preprocessing
    clean_text = preprocess(text)

    if not clean_text:
        return {
            'label'     : 'not_cyberbullying',
            'confidence': 0.0,
            'warning'   : 'Teks tidak lolos preprocessing (non-English / terlalu pendek).'
        }

    # 2. ELMo embedding → (1, max_len, 1024)
    emb = _get_elmo_embedding(clean_text, elmo, config['max_len'])

    # 3. CNN forward pass
    model = CNNInference(weights)
    probs = model.forward(emb)                   # (1, num_classes)

    # 4. Decode prediksi
    pred_idx   = int(np.argmax(probs, axis=1)[0])
    pred_label = label_enc.classes_[pred_idx]
    confidence = float(probs[0, pred_idx])

    return {
        'label'     : pred_label,
        'confidence': confidence
    }