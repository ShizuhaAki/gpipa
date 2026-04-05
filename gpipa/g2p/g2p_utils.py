from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Set

from pinyin_to_ipa import pinyin_to_ipa as _pinyin_to_ipa
from pypinyin import Style, lazy_pinyin, load_phrases_dict

try:
    from praatio import textgrid
except Exception:
    textgrid = None


DEFAULT_CUSTOM_PHRASES: Dict[str, List[str]] = {}

_LOADED_CUSTOM_PHRASES_SIGNATURES: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
_ERHUA_WORDS: Optional[Set[str]] = None
_PINYIN_TONE_RE = re.compile(r"^([a-z]+r?)([1-5]?)$")


def _normalize_custom_phrases(
    custom_phrases: Optional[Mapping[str, Iterable[str]]],
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {
        phrase: [str(syllable).strip().lower() for syllable in syllables if str(syllable).strip()]
        for phrase, syllables in DEFAULT_CUSTOM_PHRASES.items()
    }

    if not custom_phrases:
        return merged

    for phrase, syllables in custom_phrases.items():
        key = str(phrase).strip()
        if not key:
            continue
        values = [
            str(syllable).strip().lower()
            for syllable in syllables
            if str(syllable).strip()
        ]
        if values:
            merged[key] = values

    return merged


def _custom_phrase_signature(phrases: Dict[str, List[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    normalized_items = [
        (str(key), tuple(str(syllable) for syllable in value))
        for key, value in phrases.items()
        if str(key).strip() and value
    ]
    normalized_items.sort(key=lambda item: item[0])
    return tuple(normalized_items)


def _load_custom_phrases_once(
    custom_phrases: Optional[Mapping[str, Iterable[str]]] = None,
) -> None:
    phrases = _normalize_custom_phrases(custom_phrases)
    signature = _custom_phrase_signature(phrases)
    if not signature or signature in _LOADED_CUSTOM_PHRASES_SIGNATURES:
        return

    load_phrases_dict(
        {phrase: [[syllable] for syllable in syllables] for phrase, syllables in phrases.items()},
        style="tone3",
    )
    _LOADED_CUSTOM_PHRASES_SIGNATURES.add(signature)


def _normalize_phone(phone: str) -> str:
    return str(phone).strip().lower()


def _split_pinyin_tone(token: str) -> tuple[str, str]:
    m = _PINYIN_TONE_RE.match(_normalize_phone(token))
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def _set_pinyin_tone(token: str, tone: str) -> str:
    body, current_tone = _split_pinyin_tone(token)
    if not body:
        return _normalize_phone(token)

    target_tone = str(tone).strip() or current_tone
    if target_tone not in {"1", "2", "3", "4", "5"}:
        target_tone = current_tone
    return f"{body}{target_tone}"


def _apply_explicit_tone_sandhi(text: str, tokens: List[str]) -> List[str]:
    if not text or len(tokens) != len(text):
        return tokens

    out = [_normalize_phone(token) for token in tokens]

    # Normalize tone-less pinyin to neutral tone 5.
    for idx, token in enumerate(out):
        body, tone = _split_pinyin_tone(token)
        if body and not tone:
            out[idx] = f"{body}5"

    def _next_tone(idx: int) -> str:
        for j in range(idx + 1, len(out)):
            _, tone = _split_pinyin_tone(out[j])
            if tone:
                return tone
        return ""

    # 一: yi1 -> yi2 before tone4, yi4 before tone1/2/3.
    for idx, char in enumerate(text):
        if char != "一":
            continue
        body, tone = _split_pinyin_tone(out[idx])
        if not body.startswith("yi"):
            continue
        next_tone = _next_tone(idx)
        if next_tone == "4":
            out[idx] = _set_pinyin_tone(out[idx], "2")
        elif next_tone in {"1", "2", "3"}:
            out[idx] = _set_pinyin_tone(out[idx], "4")
        elif not tone:
            out[idx] = _set_pinyin_tone(out[idx], "1")

    # 不: bu4 -> bu2 before tone4.
    for idx, char in enumerate(text):
        if char != "不":
            continue
        body, _ = _split_pinyin_tone(out[idx])
        if body != "bu":
            continue
        next_tone = _next_tone(idx)
        if next_tone == "4":
            out[idx] = _set_pinyin_tone(out[idx], "2")
        else:
            out[idx] = _set_pinyin_tone(out[idx], "4")

    # Third-tone sandhi: in a 3-3 chain, all but the last become tone2.
    run_start = 0
    while run_start < len(out):
        _, tone = _split_pinyin_tone(out[run_start])
        if tone != "3":
            run_start += 1
            continue

        run_end = run_start
        while run_end + 1 < len(out):
            _, next_tone = _split_pinyin_tone(out[run_end + 1])
            if next_tone != "3":
                break
            run_end += 1

        if run_end > run_start:
            for idx in range(run_start, run_end):
                out[idx] = _set_pinyin_tone(out[idx], "2")
        run_start = run_end + 1

    return out


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
    custom_phrases: Optional[Mapping[str, Iterable[str]]] = None,
) -> List[str]:
    text = str(text)
    if not text:
        return []

    _load_custom_phrases_once(custom_phrases=custom_phrases)
    phones = lazy_pinyin(
        text,
        style=Style.TONE3,
        tone_sandhi=tone_sandhi,
        neutral_tone_with_five=neutral_tone_with_five,
        strict=False,
        errors="default",
    )

    if len(phones) == len(text):
        tokens = [_normalize_phone(phone) for phone in phones]
        if tone_sandhi:
            tokens = _apply_explicit_tone_sandhi(text, tokens)
        return _apply_vocab_normalization(text, tokens)

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
    if tone_sandhi:
        fallback = _apply_explicit_tone_sandhi(text, fallback)
    return _apply_vocab_normalization(text, fallback)


def _restore_vocab_phone_for_ipa(token: str) -> str:
    token = _normalize_phone(token)
    if not token:
        return token

    m = re.match(r"^(zh|ch|sh)iii([1-5]?)$", token)
    if m:
        return f"{m.group(1)}i{m.group(2)}"

    m = re.match(r"^(z|c|s)ii([1-5]?)$", token)
    if m:
        return f"{m.group(1)}i{m.group(2)}"

    return token


def hanzi2pinyin_tokens(
    text: str,
    *,
    tone_sandhi: bool = True,
    neutral_tone_with_five: bool = True,
    custom_phrases: Optional[Mapping[str, Iterable[str]]] = None,
) -> List[str]:
    """Convert hanzi text to tone3 pinyin tokens."""
    return _hanzi2pinyin_tokens(
        text,
        tone_sandhi=tone_sandhi,
        neutral_tone_with_five=neutral_tone_with_five,
        custom_phrases=custom_phrases,
    )


def _pinyin2ipa(phone: str) -> str:
    raw_phone = _normalize_phone(phone)
    if not raw_phone:
        return ""

    phone_for_ipa = _restore_vocab_phone_for_ipa(raw_phone)

    try:
        variants = _pinyin_to_ipa(phone_for_ipa)
    except Exception:
        return raw_phone

    if not variants:
        return raw_phone

    serialized = sorted({"".join(item) for item in variants})
    if not serialized:
        return raw_phone
    return serialized[0]


def pinyin_tokens_to_ipa_tokens(tokens: Iterable[str]) -> List[str]:
    return [_pinyin2ipa(token) for token in tokens if str(token).strip()]


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
    custom_phrases: Optional[Mapping[str, Iterable[str]]] = None,
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
            output_label = " ".join(
                _hanzi2pinyin_tokens(
                    raw_text,
                    tone_sandhi=tone_sandhi,
                    custom_phrases=custom_phrases,
                )
            )
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
    custom_phrases: Optional[Mapping[str, Iterable[str]]] = None,
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
            pinyin_tokens = _hanzi2pinyin_tokens(
                raw_text,
                tone_sandhi=tone_sandhi,
                custom_phrases=custom_phrases,
            )
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
    custom_phrases: Optional[Mapping[str, Iterable[str]]] = None,
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
            custom_phrases=custom_phrases,
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
            custom_phrases=custom_phrases,
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
    "pinyin_tokens_to_ipa_tokens",
    "pinyin_textgrid_to_ipa_textgrid",
    "read_textgrid_intervals",
    "write_textgrid_single_tier",
]
