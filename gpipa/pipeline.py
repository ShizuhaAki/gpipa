from gpipa.asr.qwen_asr import recognize
from gpipa.acoustic.alignment import export_textgrids
from gpipa.g2p.hanzi2pinyin import hanzi2pinyin
import logging
from pathlib import Path
import yaml
import json
import re

import torch
import torchaudio

logger = logging.getLogger(__name__)

_SILERO_MODEL = None
_SILERO_GET_TIMESTAMPS = None


def _load_silero_vad():
    global _SILERO_MODEL
    global _SILERO_GET_TIMESTAMPS
    if _SILERO_MODEL is not None and _SILERO_GET_TIMESTAMPS is not None:
        return _SILERO_MODEL, _SILERO_GET_TIMESTAMPS

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    _SILERO_MODEL = model
    _SILERO_GET_TIMESTAMPS = get_speech_timestamps
    return _SILERO_MODEL, _SILERO_GET_TIMESTAMPS


def _merge_close_segments(segments, max_gap_samples):
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start"] - prev["end"] <= max_gap_samples:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())
    return merged


def _split_long_segments(segments, max_chunk_samples):
    if max_chunk_samples <= 0:
        return segments

    out = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        while end - start > max_chunk_samples:
            cut = start + max_chunk_samples
            out.append({"start": start, "end": cut})
            start = cut
        if end > start:
            out.append({"start": start, "end": end})
    return out


def split_audio_with_silero(audio_path, config, output_dir):
    """
    Split audio into speech chunks with silero_vad.

    This implementation is scoped for files up to 3 minutes and returns
    ordered segment metadata with exported chunk wav paths.
    """
    vad_cfg = config["vad"]
    target_sr = 16000

    waveform, sr = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze(0).contiguous()

    model, get_speech_timestamps = _load_silero_vad()
    raw_segments = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=target_sr,
        threshold=vad_cfg["threshold"],
        min_speech_duration_ms=vad_cfg["min_speech_ms"],
        min_silence_duration_ms=vad_cfg["min_silence_ms"],
        speech_pad_ms=vad_cfg["speech_pad_ms"],
        return_seconds=False,
    )

    merged = _merge_close_segments(
        raw_segments,
        max_gap_samples=int(vad_cfg["merge_gap_sec"] * target_sr),
    )
    split_segments = _split_long_segments(
        merged,
        max_chunk_samples=int(vad_cfg["max_chunk_sec"] * target_sr),
    )

    min_keep_samples = int(vad_cfg["min_keep_sec"] * target_sr)
    kept_segments = [
        seg for seg in split_segments if (seg["end"] - seg["start"]) >= min_keep_samples
    ]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    segment_metas = []
    stem = Path(audio_path).stem
    for idx, seg in enumerate(kept_segments):
        start = seg["start"]
        end = seg["end"]
        chunk = waveform[start:end].unsqueeze(0)
        chunk_path = output_root / f"{stem}_seg_{idx:04d}.wav"
        torchaudio.save(str(chunk_path), chunk, target_sr)
        segment_metas.append(
            {
                "index": idx,
                "start_sec": start / target_sr,
                "end_sec": end / target_sr,
                "duration_sec": (end - start) / target_sr,
                "audio_path": str(chunk_path),
            }
        )

    return segment_metas

def _load_runtime_config(config_path):
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _get_audio_duration_sec(audio_path):
    info = torchaudio.info(audio_path)
    if info.sample_rate <= 0:
        waveform, sr = torchaudio.load(audio_path)
        return float(waveform.shape[-1]) / float(sr)
    return float(info.num_frames) / float(info.sample_rate)


def _load_vocab_set(config):
    vocab_path = Path(config["output_dir"]) / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    return set(vocab.keys())


def _normalize_tokens_for_alignment(pinyin_text, vocab_set):
    normalized = []
    for token in pinyin_text.split():
        base = re.sub(r"[1-5]$", "", token.strip().lower())
        if base in vocab_set:
            normalized.append(base)
    return normalized


def get_asr_transcriptions(audio_path, output_path):
    """Run ASR and return hanzi transcription text."""
    recognize(audio_path, output_path)
    return Path(output_path).read_text(encoding="utf-8").strip()

def get_acoustic_alignment(config_path, checkpoint_path, output_path, audio_path, transcript_path):
    export_textgrids(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_path,
        uttids=[],
        decode_mode='forced',
        mode='use',
        forced_viterbi_impl='regularized',
        wavpath=audio_path,
        transcript_path=transcript_path,
        compat_no_sil=True
    )
    return str(Path(output_path) / f"{Path(audio_path).stem}.TextGrid")


def _run_single_pipeline(config_path, checkpoint_path, output_path, audio_path, work_dir, seg_index):
    segment_dir = Path(work_dir) / f"seg_{seg_index:04d}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    asr_path = segment_dir / "asr.txt"
    pinyin_path = segment_dir / "pinyin.txt"
    vocab_set = _load_vocab_set(_load_runtime_config(config_path))

    acoustic_transcription = get_asr_transcriptions(audio_path, str(asr_path))
    pinyin_transcription = hanzi2pinyin(acoustic_transcription)
    normalized_tokens = _normalize_tokens_for_alignment(pinyin_transcription, vocab_set)
    if not normalized_tokens:
        logger.warning("Skip segment %s: no valid vocab token for alignment", seg_index)
        return {
            "audio_path": audio_path,
            "asr_text": acoustic_transcription,
            "pinyin_text": pinyin_transcription,
            "textgrid_path": None,
            "skipped": True,
        }

    pinyin_path.write_text(" ".join(normalized_tokens) + "\n", encoding="utf-8")

    textgrid_path = get_acoustic_alignment(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        audio_path=audio_path,
        transcript_path=str(pinyin_path),
    )
    return {
        "audio_path": audio_path,
        "asr_text": acoustic_transcription,
        "pinyin_text": " ".join(normalized_tokens),
        "textgrid_path": textgrid_path,
        "skipped": False,
    }

def pipeline(audio_file, config_path="config.yml"):
    config = _load_runtime_config(config_path)
    output_path = str(Path(config["output_dir"]) / "textgrids")
    checkpoint_path = str(Path(config["output_dir"]) / "best_model.pt")
    work_dir = str(Path(output_path) / "segments_work")

    duration_sec = _get_audio_duration_sec(audio_file)
    if duration_sec <= config["vad"]["max_single_pass_sec"]:
        logger.info("Short audio path: direct ASR->G2P->align")
        return {
            "mode": "single",
            "segments": [
                _run_single_pipeline(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    audio_path=audio_file,
                    work_dir=work_dir,
                    seg_index=0,
                )
            ],
        }

    logger.info("Long audio path: split then ASR->G2P->align per segment")
    segment_audio_dir = str(Path(work_dir) / "audio")
    segments = split_audio_with_silero(
        audio_path=audio_file,
        config=config,
        output_dir=segment_audio_dir,
    )

    results = []
    for seg in segments:
        seg_result = _run_single_pipeline(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            audio_path=seg["audio_path"],
            work_dir=work_dir,
            seg_index=seg["index"] + 1,
        )
        seg_result["start_sec"] = seg["start_sec"]
        seg_result["end_sec"] = seg["end_sec"]
        seg_result["duration_sec"] = seg["duration_sec"]
        results.append(seg_result)

    return {
        "mode": "segmented",
        "segments": results,
    }
