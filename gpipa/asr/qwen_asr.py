from __future__ import annotations

import os
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

_DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
_MODEL = None
_MODEL_KEY = None


def _resolve_model_path() -> str:
    configured = os.environ.get("GPIPA_QWEN_ASR_MODEL")
    if configured:
        return configured
    return _DEFAULT_MODEL_ID


def _resolve_device_map() -> str:
    configured = os.environ.get("GPIPA_QWEN_ASR_DEVICE")
    if configured:
        return configured
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device_map: str):
    configured = os.environ.get("GPIPA_QWEN_ASR_DTYPE")
    if configured:
        return getattr(torch, configured)

    if device_map.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def _get_model() -> Qwen3ASRModel:
    global _MODEL
    global _MODEL_KEY

    model_path = _resolve_model_path()
    device_map = _resolve_device_map()
    dtype = _resolve_dtype(device_map)
    cache_key = (model_path, device_map, str(dtype))

    if _MODEL is not None and _MODEL_KEY == cache_key:
        return _MODEL

    model_kwargs = {
        "dtype": dtype,
        "device_map": device_map,
        "max_inference_batch_size": 32,
        "max_new_tokens": 256,
    }

    _MODEL = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
    _MODEL_KEY = cache_key
    return _MODEL


def recognize(audio_path: str, output_path: str) -> str:
    model = _get_model()
    results = model.transcribe(audio_path, language="Chinese")
    text = results[0].text
    Path(output_path).write_text(text, encoding="utf-8")
    return text
