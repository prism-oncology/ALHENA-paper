"""
Xenium Explorer cell id <-> integer converters ('aaaachba-1' <-> 10001: base-16 decode + 1, so that 0 stays the mask background).
"""

import numpy as np


# Xenium Explorer id grammar: <HEX_LEN letters in 'a'..'p'> + SUFFIX
HEX_LEN: int = 8                          # number of base-16 letters in a Xenium id
HEX_BASE: int = 16                        # letters 'a'..'p' map to 0..15
SUFFIX: str = "-1"                        # literal trailing characters
EXPECTED_STR_LEN: int = HEX_LEN + len(SUFFIX)   # full string length (e.g. 10)
MAX_CELL_ID: int = HEX_BASE ** HEX_LEN    # 16**8; largest id after the +1 shift
_ORD_A: int = ord("a")


def validate_cell_id_string(cell_id_str: str) -> str | None:
    """
    None if the string is a well-formed Xenium id (HEX_LEN letters in 'a'..'p', then '-1'), else a message describing the first problem.
    """
    s = str(cell_id_str)
    if not s.endswith(SUFFIX):
        return f"missing or incorrect '{SUFFIX}' suffix"
    core = s[: -len(SUFFIX)]
    if len(core) != HEX_LEN:
        return f"expected exactly {HEX_LEN} characters before '{SUFFIX}', got {len(core)}"
    for ch in core:
        if not ("a" <= ch <= "p"):
            return f"invalid character '{ch}' (expected only 'a'..'p' before the suffix)"
    return None


def int_cell_id(cell_id_str: str) -> int:
    """
    Convert a Xenium Explorer alphabetical cell id to its (background-shifted)
    integer. E.g. int_cell_id('aaaachba-1') == 10001.
    """
    s = str(cell_id_str)
    err = validate_cell_id_string(s)
    if err is not None:
        raise ValueError(f"Unexpected Xenium cell_id format: '{s}' ({err}).")
    core = s[: -len(SUFFIX)]
    cell_id = 0
    for char in core:
        cell_id = cell_id * HEX_BASE + (ord(char) - _ORD_A)  # base-16, a-p -> 0-15
    return cell_id + 1  # +1 shift so 0 stays free for background


def str_cell_id(cell_id: int) -> str:
    """
    Inverse of `int_cell_id`: integer cell id -> Xenium alphabetical id.
    E.g. str_cell_id(10001) == 'aaaachba-1'.
    """
    cell_id = int(cell_id)
    if not (1 <= cell_id <= MAX_CELL_ID):
        raise ValueError(
            f"cell_id must be in [1, {MAX_CELL_ID}] for an {HEX_LEN}-letter Xenium id, got {cell_id}."
        )
    cell_id -= 1  # undo the +1 background shift
    coefs = []
    for _ in range(HEX_LEN):
        cell_id, coef = divmod(cell_id, HEX_BASE)
        coefs.append(coef)
    return "".join(chr(_ORD_A + coef) for coef in reversed(coefs)) + SUFFIX


def str_cell_id_init(cell_id: int) -> str:
    """
    Unshifted encoder, used at slide ingestion (`_1_process_slide.py`) before any mask exists.
    E.g. str_cell_id_init(10000) == 'aaaachba-1' == str_cell_id(10000 + 1).
    """
    cell_id = int(cell_id)
    if not (0 <= cell_id < MAX_CELL_ID):
        raise ValueError(
            f"cell_id must be in [0, {MAX_CELL_ID - 1}] for an {HEX_LEN}-letter Xenium id, got {cell_id}."
        )
    coefs = []
    for _ in range(HEX_LEN):
        cell_id, coef = divmod(cell_id, HEX_BASE)
        coefs.append(coef)
    return "".join(chr(_ORD_A + coef) for coef in reversed(coefs)) + SUFFIX


def int_cell_id_vec(cell_id_strs) -> np.ndarray:
    """
    Vectorized `int_cell_id` over an array-like of Xenium id strings -> int64
    array in [1, 16**8] (raises on any malformed id).
    """
    arr = np.asarray(cell_id_strs)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)

    # Unicode-str view so true character lengths are measured before any byte cast.
    str_arr = arr.astype(str)

    # 1) Length check (before the fixed-width byte cast)
    lengths = np.char.str_len(str_arr)
    if not np.all(lengths == EXPECTED_STR_LEN):
        bad = str_arr[lengths != EXPECTED_STR_LEN][:5].tolist()
        raise ValueError(
            f"Xenium cell_id strings must be exactly {EXPECTED_STR_LEN} characters "
            f"({HEX_LEN} letters + '{SUFFIX}'); offending examples={bad}"
        )

    # 2) Suffix check
    suffix_ok = np.char.endswith(str_arr, SUFFIX)
    if not np.all(suffix_ok):
        bad = str_arr[~suffix_ok][:5].tolist()
        raise ValueError(
            f"Xenium cell_id strings must end with '{SUFFIX}'; offending examples={bad}"
        )

    # 3) Decode the first HEX_LEN letters ('a'..'p' -> 0..15)
    as_bytes = str_arr.astype(f"S{EXPECTED_STR_LEN}")
    letters = as_bytes.view(np.uint8).reshape(len(str_arr), EXPECTED_STR_LEN)[:, :HEX_LEN]
    digits = letters.astype(np.int64) - _ORD_A
    if (digits < 0).any() or (digits >= HEX_BASE).any():
        bad_rows = np.where((digits < 0).any(axis=1) | (digits >= HEX_BASE).any(axis=1))[0][:5]
        raise ValueError(
            f"Xenium cell_id letters must be in [a-p]; offending examples="
            f"{[str_arr[i] for i in bad_rows]}"
        )

    weights = (HEX_BASE ** np.arange(HEX_LEN - 1, -1, -1)).astype(np.int64)  # 16^7 .. 16^0
    return (digits * weights).sum(axis=1) + 1  # +1 shift, matching int_cell_id
