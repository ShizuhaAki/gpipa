from typing import List, Dict
import hashlib
import os
from uuid import uuid4
import re
from torch import Tensor
import torchaudio
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from praatio import textgrid
import torch
import datasets
import logging
import json
logger = logging.getLogger(__name__)


@dataclass
class FrameBatch:
    fbanks: Tensor
    labels: Tensor
    mask: Tensor
    utt_ids: List[str]
    target_phonemes: List[List[str]]

from torch.utils.data import Dataset

class DataBakerDataset(Dataset):
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.wave = self.root / "Wave"
        self.prosody = self.root / "ProsodyLabeling"
        self.phone = self.root / "PhoneLabeling"
        self.records = self._collect_metadata()

    def _load_prosody_data(self):
        with open(self.prosody / "000001-010000.txt", "r", encoding="utf-8") as reader:
            lines = [l.strip() for l in reader]

        for i in range(0, len(lines), 2):
            l1 = lines[i]
            l2 = lines[i + 1]
            parts = l1.split()
            yield {
                "utt_id": parts[0].strip(),
                "transcription": "" if len(parts) < 2 else parts[1],
                "pinyin": l2.split(),
            }

    def _collect_metadata(self):
        phone_paths = {p.stem: p for p in self.phone.glob("*.interval")}
        wav_paths = {p.stem: p for p in self.wave.glob("*.wav")}

        records = []
        for item in self._load_prosody_data():
            utt_id = item["utt_id"]
            if utt_id in phone_paths and utt_id in wav_paths:
                records.append({
                    "utt_id": utt_id,
                    "audio_path": wav_paths[utt_id],
                    "interval_path": phone_paths[utt_id],
                    "transcription": item["transcription"],
                    "pinyin": item["pinyin"],
                })
        return records

    INITIALS = {
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", 
        "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "pl"
    }

    def _normalize_phone_label(self, label: str) -> str:
        label = str(label).strip()
        if not label: return label
        if label in {"sp", "sp1", "sp2", "sil", "sil0"}: return "sil"
        return re.sub(r"\d$", "", label)

    def _to_standard_pinyin(self, initial: str, final: str) -> str:
        
        # 1. 零声母规则 (Zero Initial)
        if not initial:
            # 处理 i 开头
            if final in ["i", "in", "ing"]:
                return "y" + final
            elif final.startswith("i"):
                if final == "iou": return "you"
                return "y" + final[1:]
            
            # 处理 u 开头
            elif final == "u":
                return "wu"
            elif final.startswith("u"):
                if final == "uei": return "wei"
                if final == "uen": return "wen"
                return "w" + final[1:]
            
            # 处理 v (ü) 开头：v -> yu, ve -> yue, van -> yuan, vn -> yun
            elif final.startswith("v"):
                return "yu" + final[1:]
                
            # 其他零声母 (a, o, e, ai, er 等) 直接返回
            else:
                return final

        # 2. 韵母简写规则 (有声母的情况下)
        if final == "iou": final = "iu"
        elif final == "uei": final = "ui"
        elif final == "uen": final = "un"

        # 3. j, q, x 遇到 ü(v) 去两点写成 u
        if initial in {"j", "q", "x"} and final.startswith("v"):
            final = "u" + final[1:]
            
        return initial + final

    def _merge_to_syllable(self, merged_phones: list) -> list:
        syllables = []
        i = 0
        while i < len(merged_phones):
            curr = merged_phones[i]
            
            # 规则 A：当前是声母，且后面还有元素
            if curr["label"] in self.INITIALS and i + 1 < len(merged_phones):
                nxt = merged_phones[i + 1]
                
                # 如果下一个是韵母 (不是 sil，也不是声母)，执行组合合并
                if nxt["label"] not in self.INITIALS and nxt["label"] != "sil":
                    standard_pinyin = self._to_standard_pinyin(curr["label"], nxt["label"])
                    syllables.append({
                        "start": curr["start"],
                        "end": nxt["end"],
                        "label": standard_pinyin
                    })
                    i += 2
                    continue
            
            # 规则 B：处理零声母韵母 (孤立的 a, i, u, ve 等)
            if curr["label"] not in self.INITIALS and curr["label"] != "sil":
                standard_pinyin = self._to_standard_pinyin("", curr["label"])
                syllables.append({
                    "start": curr["start"],
                    "end": curr["end"],
                    "label": standard_pinyin
                })
                i += 1
                continue
                
            # 规则 C：保留 sil 或者落单的异类
            syllables.append(curr)
            i += 1
            
        return syllables

    def _parse_textgrid(self, tg_path: Path):
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        tier = tg.getTier(tg.tierNames[0])

        merged_phones = []
        for entry in tier.entries:
            raw_label = str(entry.label)
            label = self._normalize_phone_label(raw_label)

            if not label: continue

            # 合并连续的同质特征
            if merged_phones and merged_phones[-1]["label"] == label:
                merged_phones[-1]["end"] = float(entry.end)
            else:
                merged_phones.append({
                    "start": float(entry.start),
                    "end": float(entry.end),
                    "label": label,
                })

        # 输出标准拼音序列
        return self._merge_to_syllable(merged_phones)
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import soundfile as sf

        rec = self.records[idx]
        audio, sr = sf.read(rec["audio_path"])

        return {
            "utt_id": rec["utt_id"],
            "audio": audio,
            "sampling_rate": sr,
            "transcription": rec["transcription"],
            "pinyin": rec["pinyin"],
            "intervals": self._parse_textgrid(rec["interval_path"]),
        }
    
def save_vocab(vocab: Dict[str, int], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as writer:
        json.dump(vocab, writer, ensure_ascii=True, indent=2)

class FrameWiseDataCollator:
    def __init__(
        self,
        label2id: Dict[str, int],
        sampling_rate: int = 16000,
        hop_length_ms: int = 10,
        cache_dir: str | Path | None = None,
    ):
        self.label2id = label2id
        self.sampling_rate = sampling_rate
        self.hop_length_s = hop_length_ms / 1000.0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, item: dict) -> str:
        audio_info = item.get("audio", {})
        source_hint = item.get("utt_id", "")
        signature = (
            f"v1|{item['utt_id']}|{source_hint}|{self.sampling_rate}|"
            f"{self.hop_length_s}|{len(item.get('intervals', []))}"
        )
        return hashlib.sha1(signature.encode("utf-8")).hexdigest()

    def _compute_features_and_labels(self, item: dict):
        waveform = torch.tensor(item["audio"]["array"]).unsqueeze(0).float()
        waveform = waveform * 32768.0

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_shift=self.hop_length_s * 1000,
            frame_length=25,
            sample_frequency=self.sampling_rate,
        )
        fbank = (fbank - fbank.mean(dim=0)) / (fbank.std(dim=0) + 1e-5)
        num_frames = fbank.shape[0]

        frame_labels = torch.full((num_frames,), fill_value=-100, dtype=torch.long)
        sequence_phonemes = []

        for interval in item["intervals"]:
            label_str = str(interval["label"])
            if label_str not in self.label2id:
                continue

            sequence_phonemes.append(label_str)

            start_frame = int(interval["start"] / self.hop_length_s)
            end_frame = int(interval["end"] / self.hop_length_s)

            start_frame = min(max(start_frame, 0), num_frames - 1)
            end_frame = min(end_frame, num_frames)

            frame_labels[start_frame:end_frame] = self.label2id[label_str]

        return fbank, frame_labels, sequence_phonemes

    def _load_or_build(self, item: dict):
        if self.cache_dir is None:
            return self._compute_features_and_labels(item)

        cache_path = self.cache_dir / f"{self._cache_key(item)}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            return cached["fbank"], cached["labels"], cached["target_phonemes"]

        fbank, frame_labels, sequence_phonemes = self._compute_features_and_labels(item)

        tmp_path = self.cache_dir / f"{cache_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        torch.save(
            {
                "fbank": fbank,
                "labels": frame_labels,
                "target_phonemes": sequence_phonemes,
            },
            tmp_path,
        )
        os.replace(tmp_path, cache_path)
        return fbank, frame_labels, sequence_phonemes

    def __call__(self, features: List[dict]) -> FrameBatch:
        fbanks_list = []
        labels_list = []
        utt_ids = []
        target_phonemes_list = []

        for item in features:
            utt_ids.append(item["utt_id"])

            fbank, frame_labels, sequence_phonemes = self._load_or_build(item)
            fbanks_list.append(fbank)
            labels_list.append(frame_labels)
            target_phonemes_list.append(sequence_phonemes)

        fbanks_padded = pad_sequence(fbanks_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        mask = (labels_padded != -100).long()

        return FrameBatch(
            fbanks=fbanks_padded,
            labels=labels_padded,
            mask=mask,
            utt_ids=utt_ids,
            target_phonemes=target_phonemes_list
        )