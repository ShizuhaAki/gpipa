from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .g2p_utils import hanzi2pinyin_tokens


def hanzi2pinyin(
    text: str,
    *,
    tone_sandhi: bool = True,
    neutral_tone_with_five: bool = True,
    separator: str = " ",
    custom_phrases: Optional[Mapping[str, Iterable[str]]] = None,
) -> str:
    """Convert hanzi text to tone3 pinyin string without TextGrid dependency in the API."""
    tokens = hanzi2pinyin_tokens(
        text,
        tone_sandhi=tone_sandhi,
        neutral_tone_with_five=neutral_tone_with_five,
        custom_phrases=custom_phrases,
    )
    return separator.join(tokens)


__all__ = ["hanzi2pinyin"]
