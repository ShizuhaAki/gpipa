from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set

from pinyin_to_ipa import pinyin_to_ipa as _pinyin_to_ipa
from pypinyin import Style, lazy_pinyin, load_phrases_dict

try:
    from praatio import textgrid
except Exception:
    textgrid = None


CUSTOM_PHRASES: Dict[str, List[str]] = {
    "你好": ["ni2", "hao3"],
    "老虎": ["lao3", "hu3"],
    "了解": ["liao3", "jie3"],
    "妈妈": ["ma1", "ma5"],
    "爸爸": ["ba4", "ba5"],
    "我们": ["wo3", "men5"],
    "什么": ["shen2", "me5"],
    "东西": ["dong1", "xi5"],
    "知道": ["zhi1", "dao5"],
    "一定": ["yi2", "ding4"],
    "不要": ["bu2", "yao4"],
}

_CUSTOM_PHRASES_LOADED = False
_ERHUA_WORDS: Optional[Set[str]] = None


def _load_custom_phrases_once() -> None:
    global _CUSTOM_PHRASES_LOADED
    if _CUSTOM_PHRASES_LOADED:
        return

    load_phrases_dict(
        {phrase: [[syllable] for syllable in syllables] for phrase, syllables in CUSTOM_PHRASES.items()},
        style="tone3",
    )
    _CUSTOM_PHRASES_LOADED = True


def _normalize_phone(phone: str) -> str:
    return str(phone).strip().lower()


def _load_erhua_words_once() -> Set[str]:
    global _ERHUA_WORDS
    if _ERHUA_WORDS is not None:
        return _ERHUA_WORDS

    erhua_path = Path(__file__).with_name("erhua.txt")
    if not erhua_path.exists():
        _ERHUA_WORDS = set()
        return _ERHUA_WORDS

    content = erhua_path.read_text(encoding="utf-8")
    _ERHUA_WORDS = {item.strip() for item in re.split(r"\s+", content) if item.strip()}
    return _ERHUA_WORDS


def _normalize_pinyin_token_for_vocab(token: str) -> str:
    token = _normalize_phone(token)
    if not token:
        return token

    m = re.match(r"^(zh|ch|sh)i([1-5]?)$", token)
    if m:
        return f"{m.group(1)}iii{m.group(2)}"

    m = re.match(r"^(z|c|s)i([1-5]?)$", token)
    if m:
        return f"{m.group(1)}ii{m.group(2)}"

    return token


def _append_erhua_suffix(token: str) -> str:
    m = re.match(r"^(.*?)([1-5]?)$", _normalize_phone(token))
    if not m:
        return _normalize_phone(token)

    body, tone = m.group(1), m.group(2)
    if body and not body.endswith("r"):
        body = f"{body}r"
    return f"{body}{tone}"


def _apply_vocab_normalization(text: str, tokens: List[str]) -> List[str]:
    normalized = [_normalize_pinyin_token_for_vocab(token) for token in tokens]
    if not text or len(normalized) != len(text):
        return normalized

    erhua_words = _load_erhua_words_once()
    if not erhua_words:
        return normalized

    merged: List[str] = list(normalized)
    for idx, char in enumerate(text):
        if char != "儿" or idx <= 0:
            continue
        if not _normalize_phone(merged[idx]).startswith("er"):
            continue

        # Reverse scan by extending the span to the left, and take the longest hit.
        matched = False
        for start in range(0, idx + 1):
            span = text[start : idx + 1]
            if len(span) <= 1:
                continue
            if span in erhua_words:
                merged[idx - 1] = _append_erhua_suffix(merged[idx - 1])
                merged[idx] = ""
                matched = True
                break

        if not matched:
            continue

    return [token for token in merged if token]


def _hanzi2pinyin_tokens(
    text: str,
    *,
    tone_sandhi: bool = True,
    neutral_tone_with_five: bool = True,
) -> List[str]:
    text = str(text)
    if not text:
        return []

    _load_custom_phrases_once()
    phones = lazy_pinyin(
        text,
        style=Style.TONE3,
        tone_sandhi=tone_sandhi,
        neutral_tone_with_five=neutral_tone_with_five,
        strict=False,
        errors="default",
    )

    if len(phones) == len(text):
        return _apply_vocab_normalization(text, [_normalize_phone(phone) for phone in phones])

    fallback: List[str] = []
    for char in text:
        current = lazy_pinyin(
            char,
            style=Style.TONE3,
            tone_sandhi=False,
            neutral_tone_with_five=neutral_tone_with_five,
            strict=False,
            errors="default",
        )
        fallback.append(_normalize_phone(current[0]) if current else _normalize_phone(char))
    return _apply_vocab_normalization(text, fallback)


def hanzi2pinyin_tokens(
    text: str,
    *,
    tone_sandhi: bool = True,
    neutral_tone_with_five: bool = True,
) -> List[str]:
    """Convert hanzi text to tone3 pinyin tokens."""
    return _hanzi2pinyin_tokens(
        text,
        tone_sandhi=tone_sandhi,
        neutral_tone_with_five=neutral_tone_with_five,
    )


def _pinyin2ipa(phone: str) -> str:
    phone = _normalize_phone(phone)
    if not phone:
        return ""

    try:
        variants = _pinyin_to_ipa(phone)
        if not variants:
            return phone
        return "".join(next(iter(variants)))
    except Exception:
        return phone


def _require_praatio() -> None:
    if textgrid is None:
        raise ImportError("praatio is required for TextGrid input/output support.")


def read_textgrid_intervals(
    input_path: Path | str,
    *,
    tier_name: Optional[str] = None,
    include_empty: bool = False,
) -> List[Dict[str, object]]:
    """Read one tier from a TextGrid file into interval dicts."""
    _require_praatio()
    tg = textgrid.openTextgrid(str(input_path), includeEmptyIntervals=include_empty)
    if not tg.tierNames:
        return []

    selected_tier_name = tier_name or tg.tierNames[0]
    if selected_tier_name not in tg.tierNames:
        raise ValueError(f"Tier '{selected_tier_name}' not found in {input_path}.")

    tier = tg.getTier(selected_tier_name)
    rows: List[Dict[str, object]] = []
    for entry in tier.entries:
        label = "" if entry.label is None else str(entry.label)
        if not include_empty and not label.strip():
            continue
        rows.append(
            {
                "start": float(entry.start),
                "end": float(entry.end),
                "text": label,
            }
        )
    return rows


def _escape_textgrid_text(text: str) -> str:
    return str(text).replace('"', '""')


def write_textgrid_single_tier(
    output_path: Path | str,
    intervals: Iterable[tuple[float, float, str]],
    *,
    tier_name: str,
    xmax: Optional[float] = None,
) -> None:
    """Write a single-tier TextGrid in long format."""
    output_path = Path(output_path)
    serialized = [(float(s), float(e), str(label)) for s, e, label in intervals]

    if xmax is None:
        xmax = max((end for _, end, _ in serialized), default=0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {xmax:.6f}",
        'tiers? <exists>',
        'size = 1',
        'item []:',
        '    item [1]:',
        '        class = "IntervalTier"',
        f'        name = "{_escape_textgrid_text(tier_name)}"',
        '        xmin = 0',
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(serialized)}",
    ]

    for idx, (start_s, end_s, label) in enumerate(serialized, start=1):
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {start_s:.6f}",
                f"            xmax = {end_s:.6f}",
                f'            text = "{_escape_textgrid_text(label)}"',
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hanzi_textgrid_to_pinyin_textgrid(
    input_path: Path | str,
    output_path: Path | str,
    *,
    input_tier_name: Optional[str] = None,
    output_tier_name: str = "pinyin",
    tone_sandhi: bool = True,
    include_empty: bool = True,
) -> None:
    """Convert hanzi labels in TextGrid intervals to tone3 pinyin labels."""
    source_intervals = read_textgrid_intervals(
        input_path,
        tier_name=input_tier_name,
        include_empty=include_empty,
    )

    output_intervals: List[tuple[float, float, str]] = []
    for item in source_intervals:
        raw_text = str(item["text"]).strip()
        if not raw_text:
            output_label = ""
        else:
            output_label = " ".join(_hanzi2pinyin_tokens(raw_text, tone_sandhi=tone_sandhi))
        output_intervals.append((float(item["start"]), float(item["end"]), output_label))

    write_textgrid_single_tier(
        output_path,
        output_intervals,
        tier_name=output_tier_name,
    )


def pinyin_textgrid_to_ipa_textgrid(
    input_path: Path | str,
    output_path: Path | str,
    *,
    input_tier_name: Optional[str] = None,
    output_tier_name: str = "ipa",
    include_empty: bool = True,
) -> None:
    """Convert pinyin labels in TextGrid intervals to IPA labels."""
    source_intervals = read_textgrid_intervals(
        input_path,
        tier_name=input_tier_name,
        include_empty=include_empty,
    )

    output_intervals: List[tuple[float, float, str]] = []
    for item in source_intervals:
        raw_text = str(item["text"]).strip()
        if not raw_text:
            output_label = ""
        else:
            pinyin_tokens = [_normalize_phone(token) for token in raw_text.split() if token.strip()]
            output_label = " ".join(_pinyin2ipa(token) for token in pinyin_tokens)
        output_intervals.append((float(item["start"]), float(item["end"]), output_label))

    write_textgrid_single_tier(
        output_path,
        output_intervals,
        tier_name=output_tier_name,
    )


def hanzi_textgrid_to_ipa_textgrid(
    input_path: Path | str,
    output_path: Path | str,
    *,
    input_tier_name: Optional[str] = None,
    output_tier_name: str = "ipa",
    tone_sandhi: bool = True,
    include_empty: bool = True,
) -> None:
    """Convert hanzi labels in TextGrid intervals directly to IPA labels."""
    source_intervals = read_textgrid_intervals(
        input_path,
        tier_name=input_tier_name,
        include_empty=include_empty,
    )

    output_intervals: List[tuple[float, float, str]] = []
    for item in source_intervals:
        raw_text = str(item["text"]).strip()
        if not raw_text:
            output_label = ""
        else:
            pinyin_tokens = _hanzi2pinyin_tokens(raw_text, tone_sandhi=tone_sandhi)
            output_label = " ".join(_pinyin2ipa(token) for token in pinyin_tokens)
        output_intervals.append((float(item["start"]), float(item["end"]), output_label))

    write_textgrid_single_tier(
        output_path,
        output_intervals,
        tier_name=output_tier_name,
    )


def convert_textgrid(
    input_path: Path | str,
    output_path: Path | str,
    *,
    source: str,
    target: str,
    input_tier_name: Optional[str] = None,
    output_tier_name: Optional[str] = None,
    tone_sandhi: bool = True,
    include_empty: bool = True,
) -> None:
    """Unified TextGrid conversion entrypoint for hanzi/pinyin/ipa labels."""
    route = (str(source).strip().lower(), str(target).strip().lower())

    if route == ("hanzi", "pinyin"):
        hanzi_textgrid_to_pinyin_textgrid(
            input_path,
            output_path,
            input_tier_name=input_tier_name,
            output_tier_name=output_tier_name or "pinyin",
            tone_sandhi=tone_sandhi,
            include_empty=include_empty,
        )
        return

    if route == ("pinyin", "ipa"):
        pinyin_textgrid_to_ipa_textgrid(
            input_path,
            output_path,
            input_tier_name=input_tier_name,
            output_tier_name=output_tier_name or "ipa",
            include_empty=include_empty,
        )
        return

    if route == ("hanzi", "ipa"):
        hanzi_textgrid_to_ipa_textgrid(
            input_path,
            output_path,
            input_tier_name=input_tier_name,
            output_tier_name=output_tier_name or "ipa",
            tone_sandhi=tone_sandhi,
            include_empty=include_empty,
        )
        return

    raise ValueError(
        "Unsupported TextGrid conversion route. "
        "Supported routes: hanzi->pinyin, pinyin->ipa, hanzi->ipa."
    )


__all__ = [
    "convert_textgrid",
    "hanzi2pinyin_tokens",
    "hanzi_textgrid_to_ipa_textgrid",
    "hanzi_textgrid_to_pinyin_textgrid",
    "pinyin_textgrid_to_ipa_textgrid",
    "read_textgrid_intervals",
    "write_textgrid_single_tier",
]
