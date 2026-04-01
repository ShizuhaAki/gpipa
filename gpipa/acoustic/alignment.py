import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from rich.logging import RichHandler
from gpipa.acoustic.dataloader import DataBakerDataset, FrameWiseDataCollator
from gpipa.acoustic.model import FrameWiseBiLSTM
from gpipa.acoustic.train import load_config

logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


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


def _extract_non_blank_spans(
    labels_1d: torch.Tensor,
    blank_id: int,
) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    start = None
    prev = None
    for i, v in enumerate(labels_1d.tolist()):
        token = int(v)
        if token == blank_id:
            if prev is not None and start is not None:
                spans.append((prev, start, i))
                prev = None
                start = None
            continue
        if prev is None:
            prev = token
            start = i
            continue
        if token != prev:
            spans.append((prev, start, i))
            prev = token
            start = i
    if prev is not None and start is not None:
        spans.append((prev, start, len(labels_1d)))
    return spans


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


def _viterbi_force_align_ctc(
    log_probs_tk: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
    chunk_size: int = -1,
    chunk_overlap: int = -1,
    start_state_radius: int = -1,
) -> torch.Tensor:
    if log_probs_tk.dim() != 2:
        raise ValueError(
            f"Expected log_probs_tk to be [T, K], got shape={tuple(log_probs_tk.shape)}"
        )
    t_len = int(log_probs_tk.size(0))
    if t_len <= 0:
        return torch.empty((0,), dtype=torch.long)
    if not target_ids:
        return torch.full((t_len,), fill_value=blank_id, dtype=torch.long)
    # Expanded CTC state sequence: [blank, p1, blank, p2, ..., blank, pN, blank]
    states: list[int] = [blank_id]
    for token_id in target_ids:
        states.append(int(token_id))
        states.append(blank_id)
    s_len = len(states)

    def _decode_one_window(
        log_probs_window_tk: torch.Tensor,
        allowed_start_states: list[int] | None,
    ) -> tuple[torch.Tensor, int]:
        win_t = int(log_probs_window_tk.size(0))
        if win_t <= 0:
            return torch.empty((0,), dtype=torch.long), 0
        neg_inf = float("-inf")
        # Tunable duration-control hyperparameters.
        base_stay_penalty = 0.15
        base_stay_penalty_blank = 0.25
        soft_max_run = 12
        soft_max_run_blank = 8
        overtime_penalty = 0.08
        overtime_penalty_blank = 0.15
        # dp[t, s, r] = best score at time t, state s, current run length bucket r
        # r is clipped to max_run_bucket so state size stays bounded.
        max_run_bucket = 40
        dp = torch.full(
            (win_t, s_len, max_run_bucket + 1),
            fill_value=neg_inf,
            dtype=log_probs_window_tk.dtype,
        )
        backptr_state = torch.full(
            (win_t, s_len, max_run_bucket + 1), fill_value=-1, dtype=torch.long
        )
        backptr_run = torch.full(
            (win_t, s_len, max_run_bucket + 1), fill_value=-1, dtype=torch.long
        )
        if allowed_start_states is None:
            start_states = [0]
            if s_len > 1:
                start_states.append(1)
        else:
            start_states = [s for s in allowed_start_states if 0 <= s < s_len]
            if not start_states:
                start_states = [0]
                if s_len > 1:
                    start_states.append(1)
        # Initialization: run length starts at 1 in any allowed state.
        for s in start_states:
            dp[0, s, 1] = log_probs_window_tk[0, states[s]]
            backptr_state[0, s, 1] = s
            backptr_run[0, s, 1] = 1
        for t in range(1, win_t):
            for s in range(s_len):
                token = states[s]
                is_blank = token == blank_id
                stay_pen = base_stay_penalty_blank if is_blank else base_stay_penalty
                soft_limit = soft_max_run_blank if is_blank else soft_max_run
                overtime_pen = overtime_penalty_blank if is_blank else overtime_penalty
                emit_score = log_probs_window_tk[t, token]
                # 1) Stay in same state, increasing run length.
                for prev_r in range(1, max_run_bucket + 1):
                    prev_score = dp[t - 1, s, prev_r]
                    if prev_score == neg_inf:
                        continue
                    new_r = min(prev_r + 1, max_run_bucket)
                    extra = max(0, prev_r - soft_limit + 1) * overtime_pen
                    cand = prev_score - stay_pen - extra + emit_score
                    if cand > dp[t, s, new_r]:
                        dp[t, s, new_r] = cand
                        backptr_state[t, s, new_r] = s
                        backptr_run[t, s, new_r] = prev_r
                # 2) Advance from previous state, resetting run length to 1.
                if s - 1 >= 0:
                    for prev_r in range(1, max_run_bucket + 1):
                        prev_score = dp[t - 1, s - 1, prev_r]
                        if prev_score == neg_inf:
                            continue
                        cand = prev_score + emit_score
                        if cand > dp[t, s, 1]:
                            dp[t, s, 1] = cand
                            backptr_state[t, s, 1] = s - 1
                            backptr_run[t, s, 1] = prev_r
                # 3) CTC skip from s-2, resetting run length to 1.
                can_skip = (
                    s - 2 >= 0 and states[s] != blank_id and states[s] != states[s - 2]
                )
                if can_skip:
                    for prev_r in range(1, max_run_bucket + 1):
                        prev_score = dp[t - 1, s - 2, prev_r]
                        if prev_score == neg_inf:
                            continue
                        cand = prev_score + emit_score
                        if cand > dp[t, s, 1]:
                            dp[t, s, 1] = cand
                            backptr_state[t, s, 1] = s - 2
                            backptr_run[t, s, 1] = prev_r
        # End in either the last blank or the last label, whichever is better.
        if s_len == 1:
            end_state = 0
            end_run = int(torch.argmax(dp[win_t - 1, 0]).item())
        else:
            last_blank = s_len - 1
            last_label = s_len - 2
            best_blank_run = int(torch.argmax(dp[win_t - 1, last_blank]).item())
            best_label_run = int(torch.argmax(dp[win_t - 1, last_label]).item())
            best_blank_score = dp[win_t - 1, last_blank, best_blank_run]
            best_label_score = dp[win_t - 1, last_label, best_label_run]
            if best_blank_score >= best_label_score:
                end_state = last_blank
                end_run = best_blank_run
            else:
                end_state = last_label
                end_run = best_label_run
        # Backtrace state path.
        state_path = torch.empty((win_t,), dtype=torch.long)
        cur_s = int(end_state)
        cur_r = int(end_run)
        for t in range(win_t - 1, -1, -1):
            state_path[t] = cur_s
            prev_s = int(backptr_state[t, cur_s, cur_r].item())
            prev_r = int(backptr_run[t, cur_s, cur_r].item())
            if prev_s < 0:
                prev_s = 0
            if prev_r < 1:
                prev_r = 1
            cur_s, cur_r = prev_s, prev_r
        aligned_ids_window = torch.tensor(
            [states[int(s.item())] for s in state_path], dtype=torch.long
        )
        end_state_idx = int(state_path[-1].item()) if win_t > 0 else 0
        return aligned_ids_window, end_state_idx

    # Degenerate setup falls back to one-shot decoding.
    if chunk_size <= 0 or chunk_size >= t_len:
        aligned_full, _ = _decode_one_window(log_probs_tk, allowed_start_states=None)
        return aligned_full
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 2)
    stride = max(1, chunk_size - chunk_overlap)
    left_keep = chunk_overlap // 2
    right_keep = chunk_overlap - left_keep
    stitched: list[torch.Tensor] = []
    start = 0
    anchor_state: int | None = None
    while start < t_len:
        end = min(start + chunk_size, t_len)
        window = log_probs_tk[start:end]
        allowed_states = None
        if anchor_state is not None:
            lo = max(0, anchor_state - max(0, start_state_radius))
            hi = min(s_len - 1, anchor_state + max(0, start_state_radius))
            allowed_states = list(range(lo, hi + 1))
        aligned_window, end_state = _decode_one_window(
            window, allowed_start_states=allowed_states
        )
        anchor_state = end_state
        keep_l = 0 if start == 0 else min(left_keep, int(aligned_window.numel()))
        keep_r = (
            0 if end == t_len else min(right_keep, int(aligned_window.numel()) - keep_l)
        )
        kept = aligned_window[keep_l : int(aligned_window.numel()) - keep_r]
        if kept.numel() > 0:
            stitched.append(kept)
        start += stride
    if not stitched:
        return torch.full((t_len,), fill_value=blank_id, dtype=torch.long)
    aligned_ids = torch.cat(stitched, dim=0)
    if int(aligned_ids.numel()) < t_len:
        pad = torch.full(
            (t_len - int(aligned_ids.numel()),),
            fill_value=int(aligned_ids[-1].item()),
            dtype=torch.long,
        )
        aligned_ids = torch.cat([aligned_ids, pad], dim=0)
    elif int(aligned_ids.numel()) > t_len:
        aligned_ids = aligned_ids[:t_len]
    return aligned_ids


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


def improved_decoding(logits, label2id, id2label):
    logits = logits / 1
    sil_id = label2id["sil"]
    logits[:, :, sil_id] += 2.0
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
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.dim() != 2:
        raise ValueError(
            f"Invalid waveform shape from {wav_path}: {tuple(waveform.shape)}"
        )
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != expected_sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, expected_sampling_rate)
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
    sil_id: int=0
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
) -> None:
    if mode not in {"test", "use"}:
        raise ValueError(f"Unsupported mode={mode}. Expected one of: test, use")
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label2id = _load_model(Path(checkpoint_path), device)
    id2label = {idx: label for label, idx in label2id.items()}
    blank_id = label2id.get("sil", 0)
    ds = DataBakerDataset(config["root_path"])
    rec_idx_by_utt = {rec["utt_id"]: idx for idx, rec in enumerate(ds.records)}
    collator = FrameWiseDataCollator(
        label2id=label2id,
        sampling_rate=config["sampling_rate"],
        hop_length_ms=config["hop_length_ms"],
        cache_dir=config.get("feature_cache_dir"),
    )
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
                log_probs = improved_decoding(logits, label2id, id2label)
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
                    log_probs_tk=log_probs[0, :num_frames],
                    target_ids=[label2id[p] for p in truth_non_blank],
                    blank_id=blank_id,
                )
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
                log_probs = improved_decoding(logits, label2id, id2label)
                truth_phonemes = [p for p in batch.target_phonemes[0] if p in label2id]
                truth_non_blank = [p for p in truth_phonemes if label2id[p] != blank_id]
                if not truth_non_blank:
                    logger.warning("Skip %s: no non-blank targets", utt_id)
                    continue
                pred_ids = _viterbi_force_align_ctc(
                    log_probs_tk=log_probs[0, :num_frames],
                    target_ids=[label2id[p] for p in truth_non_blank],
                    blank_id=blank_id
                )
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
        "--smooth-window",
        type=int,
        default=-1,
        help="Mode-smoothing window (frames) for frame decode. <=1 disables smoothing.",
    )
    parser.add_argument(
        "--viterbi-chunk-size",
        type=int,
        default=-1,
        help="Sliding-window size (frames) for forced Viterbi decode. <=0 disables chunking.",
    )
    parser.add_argument(
        "--viterbi-chunk-overlap",
        type=int,
        default=60,
        help="Overlap size (frames) between neighboring Viterbi chunks.",
    )
    parser.add_argument(
        "--viterbi-start-state-radius",
        type=int,
        default=6,
        help="Allowed state radius around previous chunk end-state when initializing next chunk.",
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
        smooth_window=args.smooth_window,
        viterbi_chunk_size=args.viterbi_chunk_size,
        viterbi_chunk_overlap=args.viterbi_chunk_overlap,
        viterbi_start_state_radius=args.viterbi_start_state_radius,
        mode=args.mode,
        wavpath=args.wavpath,
        transcript_path=args.transcript_path,
    )
