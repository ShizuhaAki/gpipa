from gpipa.asr.qwen_asr import recognize
from gpipa.acoustic.alignment import export_textgrids
from gpipa.g2p.g2p_utils import hanzi2pinyin_tokens, pinyin_tokens_to_ipa_tokens
import logging
from pathlib import Path
import yaml
import json
import re
from typing import Iterable, List, Sequence, Tuple

import torch
import torchaudio
from praatio import textgrid

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
    audio_info = getattr(torchaudio, "info", None)
    if callable(audio_info):
        info = audio_info(audio_path)
        if info.sample_rate > 0:
            return float(info.num_frames) / float(info.sample_rate)

    waveform, sr = torchaudio.load(audio_path)
    return float(waveform.shape[-1]) / float(sr)


def _load_vocab_set(config):
    vocab_path = Path(config["checkpoint_dir"]) / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    return set(vocab.keys())


def _normalize_tokens_for_alignment(pinyin_tokens: str | Iterable[str], vocab_set):
    normalized = []
    kept_tone_tokens = []

    if isinstance(pinyin_tokens, str):
        token_stream = pinyin_tokens.split()
    else:
        token_stream = [str(token) for token in pinyin_tokens]

    for token in token_stream:
        raw = token.strip().lower()
        if not raw:
            continue
        base = re.sub(r"[1-5]$", "", raw)
        if base in vocab_set:
            normalized.append(base)
            kept_tone_tokens.append(raw)
    return normalized, kept_tone_tokens


def _build_interval_tier(tier_name, intervals, max_t):
    return textgrid.IntervalTier(name=tier_name, entries=intervals, minT=0.0, maxT=max_t)


def _save_textgrid(tg, output_path):
    try:
        tg.save(str(output_path), format="long_textgrid", includeBlankSpaces=True)
    except TypeError:
        tg.save(str(output_path))


def _ensure_nonempty_intervals(intervals: Sequence[Tuple[float, float, str]], xmax: float):
    if intervals:
        return list(intervals)
    return [(0.0, float(xmax), "")]


def _read_single_tier_intervals(tg_path: str) -> Tuple[List[Tuple[float, float, str]], float]:
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    if not tg.tierNames:
        return [], 0.0

    tier = tg.getTier(tg.tierNames[0])
    intervals: List[Tuple[float, float, str]] = []
    xmax = 0.0
    for entry in tier.entries:
        start = float(entry.start)
        end = float(entry.end)
        label = "" if entry.label is None else str(entry.label)
        intervals.append((start, end, label))
        if end > xmax:
            xmax = end

    return intervals, xmax


def _segment_words(text: str) -> List[str]:
    try:
        import jieba
    except ImportError as exc:
        raise ImportError(
            "jieba is required for Word tier construction. Install it with: pip install jieba"
        ) from exc

    return [word for word in jieba.lcut(text) if str(word).strip()]


def _build_word_intervals(
    words: Sequence[str],
    word_token_counts: Sequence[int],
    phoneme_non_sil_intervals: Sequence[Tuple[float, float, str]],
    asr_text: str,
    xmax: float,
) -> List[Tuple[float, float, str]]:
    if not words or not phoneme_non_sil_intervals:
        label = asr_text.strip()
        return [(0.0, float(xmax), label)] if label else [(0.0, float(xmax), "")]

    intervals: List[Tuple[float, float, str]] = []
    cursor = 0
    total = len(phoneme_non_sil_intervals)

    for idx, word in enumerate(words):
        token_count = int(word_token_counts[idx]) if idx < len(word_token_counts) else 0
        if token_count <= 0:
            continue
        if cursor >= total:
            break

        end_idx = min(cursor + token_count - 1, total - 1)
        start = float(phoneme_non_sil_intervals[cursor][0])
        end = float(phoneme_non_sil_intervals[end_idx][1])
        label = str(word).strip()
        if label and end > start:
            intervals.append((start, end, label))
        cursor = end_idx + 1

    if intervals:
        return intervals

    label = asr_text.strip()
    return [(0.0, float(xmax), label)] if label else [(0.0, float(xmax), "")]


def _rewrite_three_tier_textgrid(
    alignment_textgrid_path: str,
    asr_text: str,
    ipa_tokens: Sequence[str],
    words: Sequence[str],
    word_token_counts: Sequence[int],
) -> None:
    base_intervals, xmax = _read_single_tier_intervals(alignment_textgrid_path)

    phoneme_intervals: List[Tuple[float, float, str]] = []
    non_sil_intervals: List[Tuple[float, float, str]] = []
    ipa_cursor = 0
    for start, end, label in base_intervals:
        raw_label = str(label).strip()
        if raw_label and raw_label.lower() != "sil":
            mapped_label = (
                str(ipa_tokens[ipa_cursor])
                if ipa_cursor < len(ipa_tokens)
                else raw_label
            )
            ipa_cursor += 1
            phoneme_intervals.append((start, end, mapped_label))
            non_sil_intervals.append((start, end, mapped_label))
        else:
            phoneme_intervals.append((start, end, raw_label or "sil"))

    sentence_label = asr_text.strip()
    sentence_intervals = [(0.0, float(xmax), sentence_label)] if sentence_label else [(0.0, float(xmax), "")]
    word_intervals = _build_word_intervals(
        words=words,
        word_token_counts=word_token_counts,
        phoneme_non_sil_intervals=non_sil_intervals,
        asr_text=asr_text,
        xmax=xmax,
    )

    tg = textgrid.Textgrid()
    tg.addTier(_build_interval_tier("Sentence", _ensure_nonempty_intervals(sentence_intervals, xmax), float(xmax)))
    tg.addTier(_build_interval_tier("Word", _ensure_nonempty_intervals(word_intervals, xmax), float(xmax)))
    tg.addTier(_build_interval_tier("Phoneme", _ensure_nonempty_intervals(phoneme_intervals, xmax), float(xmax)))
    _save_textgrid(tg, Path(alignment_textgrid_path))


def _merge_segment_textgrids(segment_results, merged_textgrid_path, total_duration_sec):
    tier_to_intervals = {}

    for seg in segment_results:
        tg_path = seg.get("textgrid_path")
        if seg.get("skipped") or not tg_path:
            continue

        seg_offset = float(seg.get("start_sec", 0.0))
        seg_tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        if not seg_tg.tierNames:
            continue

        for tier_name in seg_tg.tierNames:
            tier = seg_tg.getTier(tier_name)
            bucket = tier_to_intervals.setdefault(tier_name, [])
            for entry in tier.entries:
                start = float(entry.start) + seg_offset
                end = float(entry.end) + seg_offset
                label = "" if entry.label is None else str(entry.label)
                bucket.append((start, end, label))

    for intervals in tier_to_intervals.values():
        intervals.sort(key=lambda x: (x[0], x[1]))

    merged_tg = textgrid.Textgrid()
    if not tier_to_intervals:
        for tier_name in ("Sentence", "Word", "Phoneme"):
            merged_tg.addTier(
                _build_interval_tier(
                    tier_name=tier_name,
                    intervals=_ensure_nonempty_intervals([], float(total_duration_sec)),
                    max_t=float(total_duration_sec),
                )
            )
    else:
        preferred_order = ["Sentence", "Word", "Phoneme"]
        ordered_tiers = [name for name in preferred_order if name in tier_to_intervals]
        ordered_tiers.extend(
            [name for name in tier_to_intervals.keys() if name not in preferred_order]
        )

        for tier_name in ordered_tiers:
            merged_tg.addTier(
                _build_interval_tier(
                    tier_name=tier_name,
                    intervals=_ensure_nonempty_intervals(
                        tier_to_intervals[tier_name],
                        float(total_duration_sec),
                    ),
                    max_t=float(total_duration_sec),
                )
            )

    output_path_obj = Path(merged_textgrid_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    _save_textgrid(merged_tg, output_path_obj)
    return str(output_path_obj)


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
        wavpath=audio_path,
        transcript_path=transcript_path,
    )
    return str(Path(output_path) / f"{Path(audio_path).stem}.TextGrid")


def _run_single_pipeline(config_path, checkpoint_path, output_path, audio_path, work_dir, seg_index):
    segment_dir = Path(work_dir) / f"seg_{seg_index:04d}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    asr_path = segment_dir / "asr.txt"
    pinyin_path = segment_dir / "pinyin.txt"
    runtime_config = _load_runtime_config(config_path)
    g2p_cfg = runtime_config.get("g2p") or {}
    custom_phrases = g2p_cfg.get("custom_phrases")
    vocab_set = _load_vocab_set(runtime_config)

    acoustic_transcription = get_asr_transcriptions(audio_path, str(asr_path))
    pinyin_tokens_tone = hanzi2pinyin_tokens(
        acoustic_transcription,
        custom_phrases=custom_phrases,
    )
    normalized_tokens, aligned_tone_tokens = _normalize_tokens_for_alignment(
        pinyin_tokens_tone,
        vocab_set,
    )
    ipa_tokens = pinyin_tokens_to_ipa_tokens(aligned_tone_tokens)

    words = _segment_words(acoustic_transcription)
    word_token_counts = []
    for word in words:
        _, word_kept_tones = _normalize_tokens_for_alignment(
            hanzi2pinyin_tokens(word, custom_phrases=custom_phrases),
            vocab_set,
        )
        word_token_counts.append(len(word_kept_tones))

    if not normalized_tokens:
        logger.warning("Skip segment %s: no valid vocab token for alignment", seg_index)
        return {
            "audio_path": audio_path,
            "asr_text": acoustic_transcription,
            "pinyin_tokens_tone": pinyin_tokens_tone,
            "align_tokens_no_tone": [],
            "ipa_tokens": [],
            "alignment_textgrid_path": None,
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

    _rewrite_three_tier_textgrid(
        alignment_textgrid_path=textgrid_path,
        asr_text=acoustic_transcription,
        ipa_tokens=ipa_tokens,
        words=words,
        word_token_counts=word_token_counts,
    )

    return {
        "audio_path": audio_path,
        "asr_text": acoustic_transcription,
        "pinyin_tokens_tone": pinyin_tokens_tone,
        "align_tokens_no_tone": normalized_tokens,
        "ipa_tokens": ipa_tokens,
        "alignment_textgrid_path": textgrid_path,
        "textgrid_path": textgrid_path,
        "skipped": False,
    }

def pipeline(audio_file, config_path="config.yml"):
    config = _load_runtime_config(config_path)
    output_path = str(Path(config["output_dir"]) / "textgrids")
    checkpoint_path = str(Path(config["checkpoint_dir"]) / "best_model.pt")
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

    merged_textgrid_path = str(Path(output_path) / f"{Path(audio_file).stem}.TextGrid")
    merged_textgrid_path = _merge_segment_textgrids(
        segment_results=results,
        merged_textgrid_path=merged_textgrid_path,
        total_duration_sec=duration_sec,
    )

    return {
        "mode": "segmented",
        "segments": results,
        "merged_textgrid_path": merged_textgrid_path,
    }
