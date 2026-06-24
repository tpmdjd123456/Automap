r"""Drop non-informational candidates before greedy_partition (WP3).

A candidate is a list of (left, right) pairs from one table's column-pair. We
drop the candidate when the pair set has no mapping content worth synthesizing
across tables. Rules are intentionally narrow — they target patterns that the
advisor's "no scenario where this would be useful" test rules out.

See `is_noise(pairs)` for the predicate. Returns (drop: bool, reason: str|None).
Reasons (for reporting):
  identity              - L == R for every pair
  int->int              - every left AND every right value is numeric
  rank->rank            - both columns are consecutive integer runs (1..n)
  hex_pair              - every value on both sides matches #hex or 0xhex
  placeholder_dominated - every pair has at least one side that is a
                          placeholder token (-, --, '', n/a, na, null, none, ?)
  numeric_string        - at least one side of every pair is a number written
                          as a string (e.g. 0.5, 23,232,230, 1.5e10); years
                          (1900-2099) and date patterns (04.10, 05/07/1999,
                          06-11-2012) are excluded and kept
  currency_value        - at least one side of every pair contains a currency
                          symbol paired with a number (e.g. $9.99, 14.00€, £3)
  number_dominated      - at least one side of every pair contains both letters
                          and digits but digits outnumber letters (e.g. 14B3Z9,
                          A1B2C3D4); strings like 'Building 14' or 'Room 3A'
                          are kept; dates are excluded
  boolean_pair          - at least one side of every pair is a boolean-like
                          token (true/false, yes/no, on/off, 0/1, t/f, y/n)
  uuid_pair             - at least one side of every pair is a UUID or compact
                          32-char hex hash (e.g. a1b2c3d4-e5f6-...)
  url_or_path           - at least one side of every pair is a URL (http/https)
                          or an absolute file-system path (/usr/bin, C:/Users)
  repeated_char         - at least one side of every pair is a run of 3+
                          identical non-alphanumeric characters (e.g. ----, ....
  constant_pair         - one entire column is a single repeated value,
                          meaning the mapping carries no information
  index_offset_pair     - both columns are consecutive integer runs that may
                          be offset from each other (e.g. 0-based vs 1-based
                          index remapping); generalises rank->rank
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# A "number" for our purposes: optional sign, digits, optional single decimal
# group with `.` or `,`. Deliberately strict — we don't want to match "$1.2m"
# or "-0.0%" as numeric (those still have signal as text).
_NUM = re.compile(r"^-?\d+(?:[.,]\d+)?$")
_INT = re.compile(r"^-?\d+$")
_HEX = re.compile(r"^(?:#|0x)[0-9a-fA-F]+$")

# Tokens that mean "no value" in web tables. `0` is *not* a placeholder — it's a
# real measurement in many contexts (counts, scores).
_PLACEHOLDERS = frozenset({"", "-", "--", "—", "n/a", "na", "null", "none", "?"})

_CURRENCY_RE = re.compile(
    r'[$€£¥₹₩₺₽¢][\s]?\d|'       # symbol before number
    r'\d[\s]?[$€£¥₹₩₺₽¢]',        # symbol after number
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r'''
    (?:
        \d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?   # 04.10 | 05/07/1999 | 06-11-2012
      | \d{4}[./-]\d{1,2}[./-]\d{1,2}           # 1999-07-05 (ISO)
    )$
    ''',
    re.VERBOSE,
)

_YEAR_RE    = re.compile(r'^(19|20)\d{2}$')          # 1900–2099
_PURE_NUM_RE = re.compile(r'^[+\-]?[\d][\d ,._]*(?:e[+\-]?\d+)?$', re.IGNORECASE)
# matches: 42  0.5  23,232,230  1_000  1.5e10  +3  -0.7

_BOOL_TOKENS = {"true","false","yes","no","on","off","0","1","t","f","y","n"}
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    r'|^[0-9a-f]{32}$',           # compact form
    re.IGNORECASE,
)
_URL_RE   = re.compile(r'^https?://', re.IGNORECASE)
_PATH_RE  = re.compile(r'^(?:/[\w.\-]+)+/?$|^[A-Za-z]:\\', )
_REPEAT_RE = re.compile(r'^([^a-zA-Z0-9])\1{2,}$')   # ---- ???? ......


def _is_date_like(value: str) -> bool:
    """Return True if the value looks like a date or a standalone year."""
    v = value.strip()
    return bool(_YEAR_RE.match(v) or _DATE_RE.match(v))


def _is_pure_numeric_string(value: str) -> bool:
    """
    True when the value is a number written as a string,
    but NOT a year or date expression.
    """
    v = value.strip()
    if _is_date_like(v):
        return False
    return bool(_PURE_NUM_RE.match(v))


def _is_number_dominated(value: str) -> bool:
    """
    True when digits outnumber ASCII letters in the string,
    AND there is at least one letter (otherwise int->int catches it).
    Keeps 'Building 14', 'Room 3A'; drops '14B3Z9', 'A1B2C3D4'.
    Date-like values are excluded first.
    """
    v = value.strip()
    if _is_date_like(v):
        return False
    letters = sum(c.isalpha() for c in v)
    digits  = sum(c.isdigit() for c in v)
    # must have both kinds, and digits must dominate
    return letters > 0 and digits > letters


def _is_currency_value(value: str) -> bool:
    return bool(_CURRENCY_RE.search(value.strip()))


def _check_constant_pair(pairs):
    """One entire column is a single repeated constant."""
    lefts  = {str(l) for l, _ in pairs}
    rights = {str(r) for _, r in pairs}
    if len(lefts) == 1 or len(rights) == 1:
        return True, "constant_pair"
    return False, None

def _check_index_offset_pair(pairs):
    """L and R are both consecutive integer runs (possibly offset from each other)."""
    try:
        ls = [int(l) for l, _ in pairs]
        rs = [int(r) for _, r in pairs]
    except (ValueError, TypeError):
        return False, None
    def _is_run(seq):
        return seq == list(range(seq[0], seq[0] + len(seq)))
    if _is_run(sorted(ls)) and _is_run(sorted(rs)):
        return True, "index_offset_pair"
    return False, None



def _is_num(v: str) -> bool:
    return bool(_NUM.match(v.strip()))


def _is_int(v: str) -> bool:
    return bool(_INT.match(v.strip()))


def _is_hex(v: str) -> bool:
    return bool(_HEX.match(v.strip()))


def _is_placeholder(v: str) -> bool:
    return v.strip().lower() in _PLACEHOLDERS


def is_noise_value(v: str) -> bool:
    """True iff value matches the column-level Vertica filter at the value level.

    Used by build_cooccurrence_index and compute_coherence to skip pure-numeric /
    hex / placeholder tokens *within* surviving mixed columns. These values
    contribute near-zero NPMI signal but dominate index size at scale.
    """
    return _is_num(v) or _is_hex(v) or _is_placeholder(v)


def _is_rank_run(vals: Sequence[str]) -> bool:
    """True iff `vals` are a consecutive integer sequence (any order)
    anchored at 0 or 1, e.g. {1,2,…,n} or {0,1,…,n-1}, length >= 3."""
    if len(vals) < 3 or not all(_is_int(v) for v in vals):
        return False
    nums = sorted(int(v) for v in vals)
    if len(set(nums)) != len(nums):
        return False
    return nums[0] in (0, 1) and nums == list(range(nums[0], nums[0] + len(nums)))


def _check_numeric_string(pairs):
    if all(
        _is_pure_numeric_string(str(l)) or _is_pure_numeric_string(str(r))
        for l, r in pairs
    ):
        return True, "numeric_string"
    return False, None


def _check_currency_value(pairs):
    if all(
        _is_currency_value(str(l)) or _is_currency_value(str(r))
        for l, r in pairs
    ):
        return True, "currency_value"
    return False, None


def _check_number_dominated(pairs):
    if all(
        _is_number_dominated(str(l)) or _is_number_dominated(str(r))
        for l, r in pairs
    ):
        return True, "number_dominated"
    return False, None


def _check_boolean_pair(pairs):
    if all(
        str(l).strip().lower() in _BOOL_TOKENS or
        str(r).strip().lower() in _BOOL_TOKENS
        for l, r in pairs
    ):
        return True, "boolean_pair"
    return False, None


def _check_uuid_pair(pairs):
    if all(
        _UUID_RE.match(str(l).strip()) or _UUID_RE.match(str(r).strip())
        for l, r in pairs
    ):
        return True, "uuid_pair"
    return False, None


def _is_url_or_path_value(v):
    v = str(v).strip()
    return bool(_URL_RE.match(v) or _PATH_RE.match(v))


def _check_url_or_path(pairs):
    if all(
        _is_url_or_path_value(l) or _is_url_or_path_value(r)
        for l, r in pairs
    ):
        return True, "url_or_path"
    return False, None


def _check_repeated_char(pairs):
    if all(
        _REPEAT_RE.match(str(l).strip()) or _REPEAT_RE.match(str(r).strip())
        for l, r in pairs
    ):
        return True, "repeated_char"
    return False, None

def is_noise(pairs: Sequence[Tuple[str, str]]) -> Tuple[bool, Optional[str]]:
    """Classify a candidate's pair list. Returns (drop, reason).

    Drop iff any rule matches; first matching rule's name is the reason.
    """
    if not pairs:
        return True, "empty"

    if all(l == r for l, r in pairs):
        return True, "identity"

    lefts  = [p[0] for p in pairs]
    rights = [p[1] for p in pairs]

    # ── original rules ────────────────────────────────────────────────────────
    if all(_is_num(v) for v in lefts) and all(_is_num(v) for v in rights):
        return True, "int->int"

    if _is_rank_run(lefts) and _is_rank_run(rights):
        return True, "rank->rank"

    if all(_is_hex(v) for v in lefts) and all(_is_hex(v) for v in rights):
        return True, "hex_pair"

    if all(_is_placeholder(l) or _is_placeholder(r) for l, r in pairs):
        return True, "placeholder_dominated"

    # ── new rules ─────────────────────────────────────────────────────────────
    if all(_is_pure_numeric_string(str(l)) or _is_pure_numeric_string(str(r))
           for l, r in pairs):
        return True, "numeric_string"

    if all(_is_currency_value(str(l)) or _is_currency_value(str(r))
           for l, r in pairs):
        return True, "currency_value"

    if all(_is_number_dominated(str(l)) or _is_number_dominated(str(r))
           for l, r in pairs):
        return True, "number_dominated"

    if all(str(l).strip().lower() in _BOOL_TOKENS or
           str(r).strip().lower() in _BOOL_TOKENS
           for l, r in pairs):
        return True, "boolean_pair"

    if all(_UUID_RE.match(str(l).strip()) or _UUID_RE.match(str(r).strip())
           for l, r in pairs):
        return True, "uuid_pair"

    if all(_is_url_or_path_value(str(l)) or _is_url_or_path_value(str(r))
           for l, r in pairs):
        return True, "url_or_path"

    if all(_REPEAT_RE.match(str(l).strip()) or _REPEAT_RE.match(str(r).strip())
           for l, r in pairs):
        return True, "repeated_char"

    lefts_set  = {str(v) for v in lefts}
    rights_set = {str(v) for v in rights}
    if len(lefts_set) == 1 or len(rights_set) == 1:
        return True, "constant_pair"

    try:
        ls = sorted(int(l) for l in lefts)
        rs = sorted(int(r) for r in rights)
        if ls == list(range(ls[0], ls[0] + len(ls))) and \
           rs == list(range(rs[0], rs[0] + len(rs))):
            return True, "index_offset_pair"
    except (ValueError, TypeError):
        pass

    return False, None


def filter_candidates(candidates: List[dict]) -> Tuple[List[dict], dict]:
    """Apply `is_noise` to a list of candidate dicts (each with a "pairs" key).
    Returns (kept_candidates, drop_counts_by_reason)."""
    kept: List[dict] = []
    drops: dict = {}
    for c in candidates:
        drop, reason = is_noise(c["pairs"])
        if drop:
            drops[reason] = drops.get(reason, 0) + 1
        else:
            kept.append(c)
    return kept, drops