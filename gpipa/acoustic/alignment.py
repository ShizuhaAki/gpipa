import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from rich.logging import RichHandler
from gpipa.acoustic.dataloader import DataBakerDataset, FrameWiseDataCollator
from gpipa.acoustic.model import FrameWiseBiLSTM
from gpipa.acoustic.train import load_config

try:
    from numba import njit
except ImportError:
    njit = None

logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEFAULT_VITERBI_CONFIG = {
    "base_stay_penalty": 0.2,
    "base_stay_penalty_blank": 0.25,
    "soft_max_run": 8,
    "soft_max_run_blank": 8,
    "overtime_penalty": 0.8,
    "overtime_penalty_blank": 0.15,
    "max_run_bucket": 80,
}


def _resolve_viterbi_config(raw_cfg: dict | None) -> dict:
    cfg = dict(DEFAULT_VITERBI_CONFIG)
    if isinstance(raw_cfg, dict):
        cfg.update(raw_cfg)

    return {
        "base_stay_penalty": max(0.0, float(cfg["base_stay_penalty"])),
        "base_stay_penalty_blank": max(0.0, float(cfg["base_stay_penalty_blank"])),
        "soft_max_run": max(1, int(cfg["soft_max_run"])),
        "soft_max_run_blank": max(1, int(cfg["soft_max_run_blank"])),
        "overtime_penalty": max(0.0, float(cfg["overtime_penalty"])),
        "overtime_penalty_blank": max(0.0, float(cfg["overtime_penalty_blank"])),
        "max_run_bucket": max(2, int(cfg["max_run_bucket"])),
    }


def _sanitize_omp_num_threads() -> None:
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return
    value = raw.strip()
    if value.isdigit() and int(value) >= 1:
        os.environ["OMP_NUM_THREADS"] = value
        return
    logger.warning("Invalid OMP_NUM_THREADS=%r. Falling back to 1.", raw)
    os.environ["OMP_NUM_THREADS"] = "1"


def _load_model(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "label2id" not in checkpoint:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain label2id")
    label2id: dict[str, int] = checkpoint["label2id"]
    ckpt_cfg = checkpoint.get("config") or {}
    model = FrameWiseBiLSTM(
        input_dim=ckpt_cfg.get("input_dim", 80),
        hidden_dim=ckpt_cfg.get("hidden_dim", 256),
        num_layers=ckpt_cfg.get("num_layers", 3),
        num_classes=len(label2id),
        dropout=ckpt_cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label2id


def _normalize_item(item: dict) -> dict:
    if isinstance(item.get("audio"), dict) and "array" in item["audio"]:
        return item
    new_item = dict(item)
    new_item["audio"] = {"array": item["audio"]}
    return new_item


def _extract_spans_from_ids(labels_1d: torch.Tensor) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    if labels_1d.numel() == 0:
        return spans
    start = 0
    prev = int(labels_1d[0].item())
    for i in range(1, int(labels_1d.numel())):
        cur = int(labels_1d[i].item())
        if cur != prev:
            spans.append((prev, start, i))
            start = i
            prev = cur
    spans.append((prev, start, int(labels_1d.numel())))
    return spans


def _viterbi_force_align_ctc_numpy_impl(
    log_probs_tk: np.ndarray,
    states: np.ndarray,
    blank_id: int,
    base_stay_penalty: float,
    base_stay_penalty_blank: float,
    soft_max_run: int,
    soft_max_run_blank: int,
    overtime_penalty: float,
    overtime_penalty_blank: float,
    max_run_bucket: int,
) -> np.ndarray:
    win_t = int(log_probs_tk.shape[0])
    s_len = int(states.shape[0])
    if win_t <= 0:
        return np.empty((0,), dtype=np.int64)

    neg_inf = np.float32(-1e30)
    base_stay_penalty = np.float32(base_stay_penalty)
    base_stay_penalty_blank = np.float32(base_stay_penalty_blank)
    soft_max_run = int(soft_max_run)
    soft_max_run_blank = int(soft_max_run_blank)
    overtime_penalty = np.float32(overtime_penalty)
    overtime_penalty_blank = np.float32(overtime_penalty_blank)
    max_run_bucket = int(max_run_bucket)

    dp = np.full((win_t, s_len, max_run_bucket + 1), neg_inf, dtype=np.float32)
    backptr_state = np.full((win_t, s_len, max_run_bucket + 1), -1, dtype=np.int32)
    backptr_run = np.full((win_t, s_len, max_run_bucket + 1), -1, dtype=np.int32)

    dp[0, 0, 1] = log_probs_tk[0, states[0]]
    backptr_state[0, 0, 1] = 0
    backptr_run[0, 0, 1] = 1
    if s_len > 1:
        dp[0, 1, 1] = log_probs_tk[0, states[1]]
        backptr_state[0, 1, 1] = 1
        backptr_run[0, 1, 1] = 1

    for t in range(1, win_t):
        for s in range(s_len):
            token = states[s]
            is_blank = token == blank_id
            stay_pen = base_stay_penalty_blank if is_blank else base_stay_penalty
            soft_limit = soft_max_run_blank if is_blank else soft_max_run
            overtime_pen = overtime_penalty_blank if is_blank else overtime_penalty
            emit_score = log_probs_tk[t, token]

            for prev_r in range(1, max_run_bucket + 1):
                prev_score = dp[t - 1, s, prev_r]
                if prev_score <= neg_inf / 2.0:
                    continue
                new_r = prev_r + 1
                if new_r > max_run_bucket:
                    new_r = max_run_bucket
                extra = prev_r - soft_limit + 1
                if extra < 0:
                    extra = 0
                cand = prev_score - stay_pen - extra * overtime_pen + emit_score
                if cand > dp[t, s, new_r]:
                    dp[t, s, new_r] = cand
                    backptr_state[t, s, new_r] = s
                    backptr_run[t, s, new_r] = prev_r

            if s - 1 >= 0:
                for prev_r in range(1, max_run_bucket + 1):
                    prev_score = dp[t - 1, s - 1, prev_r]
                    if prev_score <= neg_inf / 2.0:
                        continue
                    cand = prev_score + emit_score
                    if cand > dp[t, s, 1]:
                        dp[t, s, 1] = cand
                        backptr_state[t, s, 1] = s - 1
                        backptr_run[t, s, 1] = prev_r

            can_skip = s - 2 >= 0 and states[s] != blank_id and states[s] != states[s - 2]
            if can_skip:
                for prev_r in range(1, max_run_bucket + 1):
                    prev_score = dp[t - 1, s - 2, prev_r]
                    if prev_score <= neg_inf / 2.0:
                        continue
                    cand = prev_score + emit_score
                    if cand > dp[t, s, 1]:
                        dp[t, s, 1] = cand
                        backptr_state[t, s, 1] = s - 2
                        backptr_run[t, s, 1] = prev_r

    if s_len == 1:
        end_state = 0
        end_run = 1
        best_score = dp[win_t - 1, 0, 1]
        for r in range(2, max_run_bucket + 1):
            cand = dp[win_t - 1, 0, r]
            if cand > best_score:
                best_score = cand
                end_run = r
    else:
        last_blank = s_len - 1
        last_label = s_len - 2

        best_blank_run = 1
        best_blank_score = dp[win_t - 1, last_blank, 1]
        for r in range(2, max_run_bucket + 1):
            cand = dp[win_t - 1, last_blank, r]
            if cand > best_blank_score:
                best_blank_score = cand
                best_blank_run = r

        best_label_run = 1
        best_label_score = dp[win_t - 1, last_label, 1]
        for r in range(2, max_run_bucket + 1):
            cand = dp[win_t - 1, last_label, r]
            if cand > best_label_score:
                best_label_score = cand
                best_label_run = r

        if best_blank_score >= best_label_score:
            end_state = last_blank
            end_run = best_blank_run
        else:
            end_state = last_label
            end_run = best_label_run

    state_path = np.empty((win_t,), dtype=np.int32)
    cur_s = end_state
    cur_r = end_run
    for t in range(win_t - 1, -1, -1):
        state_path[t] = cur_s
        prev_s = backptr_state[t, cur_s, cur_r]
        prev_r = backptr_run[t, cur_s, cur_r]
        if prev_s < 0:
            prev_s = 0
        if prev_r < 1:
            prev_r = 1
        cur_s = prev_s
        cur_r = prev_r

    aligned_ids = np.empty((win_t,), dtype=np.int64)
    for t in range(win_t):
        aligned_ids[t] = states[state_path[t]]
    return aligned_ids


if njit is not None:
    _viterbi_force_align_ctc_numpy_impl = njit(cache=False)(
        _viterbi_force_align_ctc_numpy_impl
    )


def _viterbi_force_align_ctc(
    log_probs_tk: np.ndarray,
    target_ids: list[int],
    blank_id: int,
    viterbi_cfg: dict | None = None,
) -> np.ndarray:
    log_probs_np = np.asarray(log_probs_tk, dtype=np.float32)
    if log_probs_np.ndim != 2:
        raise ValueError(
            f"Expected log_probs_tk to be [T, K], got shape={tuple(log_probs_np.shape)}"
        )
    t_len = int(log_probs_np.shape[0])
    if t_len <= 0:
        return np.empty((0,), dtype=np.int64)
    if not target_ids:
        return np.full((t_len,), fill_value=int(blank_id), dtype=np.int64)

    targets = np.asarray(target_ids, dtype=np.int64)
    states = np.empty((targets.size * 2 + 1,), dtype=np.int64)
    states[0] = int(blank_id)
    states[1::2] = targets
    states[2::2] = int(blank_id)

    if int(states.min()) < 0 or int(states.max()) >= int(log_probs_np.shape[1]):
        raise ValueError(
            "CTC states contain token ids outside log-prob vocabulary range: "
            f"min={int(states.min())}, max={int(states.max())}, K={int(log_probs_np.shape[1])}"
        )

    cfg = _resolve_viterbi_config(viterbi_cfg)
    return _viterbi_force_align_ctc_numpy_impl(
        log_probs_np,
        states,
        int(blank_id),
        float(cfg["base_stay_penalty"]),
        float(cfg["base_stay_penalty_blank"]),
        int(cfg["soft_max_run"]),
        int(cfg["soft_max_run_blank"]),
        float(cfg["overtime_penalty"]),
        float(cfg["overtime_penalty_blank"]),
        int(cfg["max_run_bucket"]),
    )


def _escape_textgrid_text(s: str) -> str:
    return s.replace('"', '""')


def _write_textgrid_single_tier(
    output_path: Path,
    intervals: list[tuple[float, float, str]],
    xmax: float,
    tier_name: str = "phones",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {xmax:.6f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        f'        name = "{_escape_textgrid_text(tier_name)}"',
        "        xmin = 0",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for i, (start_s, end_s, label) in enumerate(intervals, start=1):
        lines.extend(
            [
                f"        intervals [{i}]:",
                f"            xmin = {start_s:.6f}",
                f"            xmax = {end_s:.6f}",
                f'            text = "{_escape_textgrid_text(label)}"',
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_uttids(uttids_raw: str | None, uttids_file: str | None) -> list[str]:
    uttids: list[str] = []
    if uttids_raw:
        uttids.extend([x.strip() for x in uttids_raw.split(",") if x.strip()])
    if uttids_file:
        path = Path(uttids_file)
        if not path.exists():
            raise FileNotFoundError(f"uttids file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        uttids.extend([line.strip() for line in lines if line.strip()])
    # Keep order but deduplicate.
    seen = set()
    deduped = []
    for utt in uttids:
        if utt not in seen:
            deduped.append(utt)
            seen.add(utt)
    return deduped


def _audio_array_to_mono_waveform(audio_array) -> torch.Tensor:
    waveform = torch.as_tensor(audio_array, dtype=torch.float32)
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() == 2:
        # Support both [C, T] and [T, C] layouts.
        if int(waveform.size(0)) <= int(waveform.size(1)):
            return waveform.mean(dim=0)
        return waveform.mean(dim=1)
    raise ValueError(f"Unsupported waveform shape: {tuple(waveform.shape)}")


def _resample_1d_to_count(values: torch.Tensor, target_count: int) -> torch.Tensor:
    if target_count <= 0:
        return torch.empty((0,), dtype=torch.float32)
    if values.numel() <= 0:
        return torch.zeros((target_count,), dtype=torch.float32)
    if int(values.numel()) == target_count:
        return values.to(dtype=torch.float32)
    if int(values.numel()) == 1:
        return values.repeat(target_count).to(dtype=torch.float32)
    x = values.view(1, 1, -1).to(dtype=torch.float32)
    y = F.interpolate(x, size=target_count, mode="linear", align_corners=True)
    return y.view(-1)


def _compute_sil_high_energy_mask(
    waveform_1d: torch.Tensor,
    sampling_rate: int,
    hop_length_ms: int,
    target_num_frames: int,
    noise_floor_quantile: float,
    noise_threshold_db: float,
) -> torch.Tensor:
    if target_num_frames <= 0:
        return torch.empty((0,), dtype=torch.bool)
    if waveform_1d.numel() <= 0:
        return torch.zeros((target_num_frames,), dtype=torch.bool)

    sr = max(1, int(sampling_rate))
    hop_samples = max(1, int(round(sr * (float(hop_length_ms) / 1000.0))))
    frame_samples = max(1, int(round(sr * 0.025)))

    wave = waveform_1d.to(dtype=torch.float32)
    if int(wave.numel()) < frame_samples:
        wave = F.pad(wave, (0, frame_samples - int(wave.numel())))

    frame_chunks = wave.unfold(0, frame_samples, hop_samples)
    frame_rms = torch.sqrt(torch.mean(frame_chunks * frame_chunks, dim=1) + 1e-12)
    frame_rms = _resample_1d_to_count(frame_rms, target_num_frames)

    q = min(1.0, max(0.0, float(noise_floor_quantile)))
    noise_floor = float(torch.quantile(frame_rms, q).item())
    noise_floor = max(noise_floor, 1e-8)
    threshold = noise_floor * (10.0 ** (float(noise_threshold_db) / 20.0))
    return frame_rms > threshold


def improved_decoding(
    logits,
    label2id,
    id2label,
    sil_high_energy_mask: torch.Tensor | None = None,
    sil_energy_penalty: float = 12.0,
):
    logits = logits / 1
    sil_id = label2id["sil"]
    logits[:, :, sil_id] += 2.0
    if sil_high_energy_mask is not None and sil_energy_penalty > 0:
        if sil_high_energy_mask.dim() == 1:
            sil_high_energy_mask = sil_high_energy_mask.unsqueeze(0)
        sil_mask = sil_high_energy_mask.to(device=logits.device, dtype=logits.dtype)
        if int(sil_mask.size(1)) != int(logits.size(1)):
            sil_mask = F.interpolate(
                sil_mask.unsqueeze(1),
                size=int(logits.size(1)),
                mode="nearest",
            ).squeeze(1)
        if int(sil_mask.size(0)) == 1 and int(logits.size(0)) > 1:
            sil_mask = sil_mask.expand(int(logits.size(0)), -1)
        logits[:, :, sil_id] = logits[:, :, sil_id] - sil_mask * float(
            sil_energy_penalty
        )
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs


def _load_transcript_tokens(transcript_path: str | Path) -> list[str]:
    path = Path(transcript_path)
    if not path.exists():
        raise FileNotFoundError(f"transcript file not found: {path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return []
    # Prefer pinyin/phoneme line when transcript is in two-line format.
    if len(lines) >= 2 and len(lines[1].split()) > 1:
        return lines[1].split()
    tokens = lines[0].split()
    if len(tokens) > 1 and tokens[0].isdigit():
        return tokens[1:]
    return tokens


def _wav_to_fbank(
    wav_path: str | Path,
    expected_sampling_rate: int,
    hop_length_ms: int,
) -> torch.Tensor:
    waveform_loaded, loaded_sr = torchaudio.load(str(wav_path))
    waveform_1d = _audio_array_to_mono_waveform(waveform_loaded)
    if int(loaded_sr) != int(expected_sampling_rate):
        waveform_1d = torchaudio.functional.resample(
            waveform_1d.unsqueeze(0), int(loaded_sr), int(expected_sampling_rate)
        ).squeeze(0)
    waveform = waveform_1d.unsqueeze(0)
    waveform = waveform.float() * 32768.0
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_shift=hop_length_ms,
        frame_length=25,
        sample_frequency=expected_sampling_rate,
    )
    fbank = (fbank - fbank.mean(dim=0)) / (fbank.std(dim=0) + 1e-5)
    return fbank


def _ids_to_intervals(
    ids_1d: torch.Tensor,
    id2label: dict[int, str],
    hop_s: float,
) -> tuple[list[tuple[float, float, str]], float]:
    spans_all = _extract_spans_from_ids(ids_1d)
    intervals: list[tuple[float, float, str]] = []
    for token, start_f, end_f in spans_all:
        if end_f <= start_f:
            continue
        intervals.append((start_f * hop_s, end_f * hop_s, id2label.get(token, "<unk>")))
    xmax = int(ids_1d.numel()) * hop_s
    return intervals, xmax


def _extract_change_boundaries(ids_1d: torch.Tensor) -> torch.Tensor:
    """Return frame indices where label transitions happen."""
    if ids_1d.numel() <= 1:
        return torch.empty((0,), dtype=torch.float32)
    changes = ids_1d[1:] != ids_1d[:-1]
    boundaries = torch.nonzero(changes, as_tuple=False).view(-1) + 1
    return boundaries.to(dtype=torch.float32)


def _resample_boundaries_to_count(
    boundaries: torch.Tensor, target_count: int
) -> torch.Tensor:
    if target_count <= 0:
        return torch.empty((0,), dtype=torch.float32)
    if boundaries.numel() <= 0:
        return torch.zeros((target_count,), dtype=torch.float32)
    if boundaries.numel() == target_count:
        return boundaries.to(dtype=torch.float32)
    if boundaries.numel() == 1:
        return boundaries.repeat(target_count).to(dtype=torch.float32)
    x = boundaries.view(1, 1, -1).to(dtype=torch.float32)
    y = F.interpolate(x, size=target_count, mode="linear", align_corners=True)
    return y.view(-1)


def _boundary_time_mae_ms(
    pred_ids: torch.Tensor, gt_ids: torch.Tensor, hop_s: float
) -> float | None:
    """Compute boundary MAE in milliseconds for frame-level alignments."""
    pred_b = _extract_change_boundaries(pred_ids)
    gt_b = _extract_change_boundaries(gt_ids)
    if gt_b.numel() == 0:
        return None
    pred_b_aligned = _resample_boundaries_to_count(
        pred_b, target_count=int(gt_b.numel())
    )
    mae_frames = torch.mean(torch.abs(pred_b_aligned - gt_b)).item()
    return float(mae_frames * hop_s * 1000.0)


def _redistribute_short_internal_sil(
    ids_1d: torch.Tensor,
    sil_id: int=0,
) -> torch.Tensor:
    """Split short internal sil spans equally to left/right neighbors."""
    out = ids_1d.clone()
    spans = _extract_spans_from_ids(out)
    if len(spans) < 3:
        return out

    for i, (token, start_f, end_f) in enumerate(spans):
        if token != sil_id:
            continue
        if i == 0 or i == len(spans) - 1:
            continue

        span_len = end_f - start_f
        if span_len <= 0:
            continue

        left_token = spans[i - 1][0]
        right_token = spans[i + 1][0]
        if left_token == sil_id or right_token == sil_id:
            continue

        left_len = span_len // 2
        right_start = start_f + left_len
        out[start_f:right_start] = left_token
        out[right_start:end_f] = right_token

    return out

def export_textgrids(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    uttids: list[str],
    decode_mode: str,
    mode: str,
    wavpath: str | None,
    transcript_path: str | None,
    sil_energy_penalty: float = 12.0,
    sil_noise_floor_quantile: float = 0.2,
    sil_energy_threshold_db: float = 6.0,
    should_compress_sil: bool = True,
) -> None:
    if mode not in {"test", "use"}:
        raise ValueError(f"Unsupported mode={mode}. Expected one of: test, use")
    config = load_config(config_path)
    alignment_cfg = config.get("alignment") or {}
    viterbi_cfg = _resolve_viterbi_config(alignment_cfg.get("viterbi"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label2id = _load_model(Path(checkpoint_path), device)
    id2label = {idx: label for label, idx in label2id.items()}
    blank_id = label2id.get("sil", 0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hop_s = float(config["hop_length_ms"]) / 1000.0
    if mode == "use":
        if not wavpath or not transcript_path:
            raise ValueError("Use mode requires both --wavpath and --transcript-path.")
        wav_path_obj = Path(wavpath)
        if not wav_path_obj.exists():
            raise FileNotFoundError(f"wav file not found: {wav_path_obj}")
        transcript_tokens = _load_transcript_tokens(transcript_path)
        with torch.no_grad():
            fbank = _wav_to_fbank(
                wav_path=wav_path_obj,
                expected_sampling_rate=int(config["sampling_rate"]),
                hop_length_ms=int(config["hop_length_ms"]),
            )
            if int(fbank.size(0)) <= 0:
                raise ValueError(f"No frames extracted from wav: {wav_path_obj}")
            logits = model(fbank.unsqueeze(0).to(device))
            num_frames = int(fbank.size(0))
            if decode_mode == "frame":
                pred_ids = logits[0, :num_frames].argmax(dim=-1).to("cpu")
            else:
                waveform_loaded, loaded_sr = torchaudio.load(str(wav_path_obj))
                waveform_1d = _audio_array_to_mono_waveform(waveform_loaded)
                expected_sr = int(config["sampling_rate"])
                if int(loaded_sr) != expected_sr:
                    waveform_1d = torchaudio.functional.resample(
                        waveform_1d.unsqueeze(0), int(loaded_sr), expected_sr
                    ).squeeze(0)
                sil_mask = _compute_sil_high_energy_mask(
                    waveform_1d=waveform_1d,
                    sampling_rate=int(config["sampling_rate"]),
                    hop_length_ms=int(config["hop_length_ms"]),
                    target_num_frames=num_frames,
                    noise_floor_quantile=sil_noise_floor_quantile,
                    noise_threshold_db=sil_energy_threshold_db,
                )
                log_probs = improved_decoding(
                    logits,
                    label2id,
                    id2label,
                    sil_high_energy_mask=sil_mask,
                    sil_energy_penalty=sil_energy_penalty,
                )
                truth_non_blank = [
                    p
                    for p in transcript_tokens
                    if p in label2id and label2id[p] != blank_id
                ]
                if not truth_non_blank:
                    raise ValueError(
                        "No valid non-blank phoneme tokens found in transcript."
                    )
                pred_ids = _viterbi_force_align_ctc(
                    log_probs_tk=log_probs[0, :num_frames].detach().cpu().numpy(),
                    target_ids=[label2id[p] for p in truth_non_blank],
                    blank_id=blank_id,
                    viterbi_cfg=viterbi_cfg,
                )
                pred_ids = torch.from_numpy(pred_ids)
            if should_compress_sil:
                pred_ids = _redistribute_short_internal_sil(
                    pred_ids,
                    sil_id=0,
                )
            intervals, xmax = _ids_to_intervals(
                pred_ids, id2label=id2label, hop_s=hop_s
            )
            out_tg = output_dir / f"{wav_path_obj.stem}.TextGrid"
            _write_textgrid_single_tier(
                out_tg, intervals, xmax=xmax, tier_name="phones"
            )
            logger.info("Exported %s", out_tg)
        return
    if not uttids:
        raise ValueError("Test mode requires uttids. Use --uttids or --uttids-file.")
    ds = DataBakerDataset(config["root_path"])
    rec_idx_by_utt = {rec["utt_id"]: idx for idx, rec in enumerate(ds.records)}
    collator = FrameWiseDataCollator(
        label2id=label2id,
        sampling_rate=config["sampling_rate"],
        hop_length_ms=config["hop_length_ms"],
        cache_dir=config.get("feature_cache_dir"),
    )
    exported = 0
    boundary_mae_ms_sum = 0.0
    boundary_mae_ms_count = 0
    with torch.no_grad():
        for utt_id in uttids:
            if utt_id not in rec_idx_by_utt:
                logger.warning("Skip %s: uttid not found in dataset metadata", utt_id)
                continue
            idx = rec_idx_by_utt[utt_id]
            item = _normalize_item(ds[idx])
            batch = collator([item])
            fbanks = batch.fbanks.to(device)
            logits = model(fbanks)
            valid_mask = batch.labels[0] != -100
            num_frames = int(valid_mask.sum().item())
            if num_frames <= 0:
                logger.warning("Skip %s: zero valid frames", utt_id)
                continue
            if decode_mode == "frame":
                pred_ids = logits[0, :num_frames].argmax(dim=-1).to("cpu")
            else:
                waveform_1d = _audio_array_to_mono_waveform(item["audio"]["array"])
                sil_mask = _compute_sil_high_energy_mask(
                    waveform_1d=waveform_1d,
                    sampling_rate=int(config["sampling_rate"]),
                    hop_length_ms=int(config["hop_length_ms"]),
                    target_num_frames=num_frames,
                    noise_floor_quantile=sil_noise_floor_quantile,
                    noise_threshold_db=sil_energy_threshold_db,
                )
                log_probs = improved_decoding(
                    logits,
                    label2id,
                    id2label,
                    sil_high_energy_mask=sil_mask,
                    sil_energy_penalty=sil_energy_penalty,
                )
                truth_phonemes = [p for p in batch.target_phonemes[0] if p in label2id]
                truth_non_blank = [p for p in truth_phonemes if label2id[p] != blank_id]
                if not truth_non_blank:
                    logger.warning("Skip %s: no non-blank targets", utt_id)
                    continue
                pred_ids = _viterbi_force_align_ctc(
                    log_probs_tk=log_probs[0, :num_frames].detach().cpu().numpy(),
                    target_ids=[label2id[p] for p in truth_non_blank],
                    blank_id=blank_id,
                    viterbi_cfg=viterbi_cfg,
                )
                pred_ids = torch.from_numpy(pred_ids)
            if should_compress_sil:
                pred_ids = _redistribute_short_internal_sil(
                    pred_ids,
                    sil_id=0,
                )
            intervals, xmax = _ids_to_intervals(
                pred_ids, id2label=id2label, hop_s=hop_s
            )
            textgrid_path = output_dir / f"{utt_id}.TextGrid"
            _write_textgrid_single_tier(
                textgrid_path, intervals, xmax=xmax, tier_name="phones"
            )
            exported += 1
            gt_ids = batch.labels[0, :num_frames].to("cpu")
            boundary_mae_ms = _boundary_time_mae_ms(
                pred_ids=pred_ids, gt_ids=gt_ids, hop_s=hop_s
            )
            if boundary_mae_ms is not None:
                boundary_mae_ms_sum += boundary_mae_ms
                boundary_mae_ms_count += 1
                logger.info("Boundary-MAE[%s] = %.2f ms", utt_id, boundary_mae_ms)
            logger.info("Exported %s", textgrid_path)
    logger.info("Done. Exported %d / %d requested uttids.", exported, len(uttids))
    if boundary_mae_ms_count > 0:
        logger.info(
            "Done. Mean Boundary-MAE = %.2f ms over %d utterances.",
            boundary_mae_ms_sum / boundary_mae_ms_count,
            boundary_mae_ms_count,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export TextGrid files for given uttids"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "use"],
        default="test",
        help="test: dataset-based export with MAE; use: direct wav+transcript inference.",
    )
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--checkpoint", type=str, default="output/checkpoints/best_model.pt"
    )
    parser.add_argument("--output-dir", type=str, default="output/textgrids")
    parser.add_argument(
        "--uttids",
        type=str,
        default=None,
        help="Comma-separated uttids, e.g. 000001,000002",
    )
    parser.add_argument(
        "--uttids-file",
        type=str,
        default=None,
        help="Path to a text file with one uttid per line",
    )
    parser.add_argument(
        "--wavpath",
        type=str,
        default=None,
        help="Path to input wav file (use mode only).",
    )
    parser.add_argument(
        "--transcript-path",
        type=str,
        default=None,
        help="Path to transcript/phoneme file (use mode only).",
    )
    parser.add_argument(
        "--decode-mode",
        type=str,
        choices=["frame", "forced"],
        default="frame",
        help="Decoding mode. 'forced' uses CTC-style Viterbi with dataloader phoneme targets.",
    )
    parser.add_argument(
        "--sil-energy-penalty",
        type=float,
        default=2.0,
        help="Penalty subtracted from sil logits on frames above the energy threshold.",
    )
    parser.add_argument(
        "--sil-noise-floor-quantile",
        type=float,
        default=0.2,
        help="Quantile used to estimate background noise floor from frame RMS energy.",
    )
    parser.add_argument(
        "--sil-energy-threshold-db",
        type=float,
        default=3.0,
        help="Frame is treated as non-noise when RMS exceeds noise floor by this many dB.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=-1,
        help="Mode-smoothing window (frames) for frame decode. <=1 disables smoothing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _sanitize_omp_num_threads()
    args = parse_args()
    uttids = _parse_uttids(args.uttids, args.uttids_file)
    export_textgrids(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        uttids=uttids,
        decode_mode=args.decode_mode,
        mode=args.mode,
        wavpath=args.wavpath,
        transcript_path=args.transcript_path,
        sil_energy_penalty=args.sil_energy_penalty,
        sil_noise_floor_quantile=args.sil_noise_floor_quantile,
        sil_energy_threshold_db=args.sil_energy_threshold_db,
        should_compress_sil=True
    )
