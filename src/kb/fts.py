"""FTS-only text normalization, separate from canonical chunk text."""

import re

# Bump when normalization changes to trigger an FTS-only rebuild.
FTS_NORMALIZATION_VERSION = "1"

# CJK letters and ideographs, including common extension/compatibility blocks.
# Exclude punctuation so whitespace around separators stays intact.
_CJK_CHAR = (
    r"\u3041-\u3096\u309d-\u309f"  # Hiragana
    r"\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff\uff66-\uff9f"  # Katakana
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # Ideographs + compatibility
    r"\U00020000-\U0002fa1f\U00030000-\U000323af"  # Supplementary ideographs
)
_CJK_WHITESPACE = re.compile(rf"(?<=[{_CJK_CHAR}])\s+(?=[{_CJK_CHAR}])")


def normalize_text_for_fts(text: str) -> str:
    """Remove layout whitespace only between adjacent CJK characters."""
    return _CJK_WHITESPACE.sub("", text)
