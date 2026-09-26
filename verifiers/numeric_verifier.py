import re
from fractions import Fraction
from typing import Optional, Union

def extract_numeric_scalar(text: Union[str, int, float]) -> Optional[float]:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    
    # Pre-clean string
    s = str(text).strip()
    s = s.replace(',', '')
    s = re.sub(r'[\$€£%]', '', s)
    
    # Try finding boxed answer e.g. \boxed{42}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', s)
    if boxed_match:
        s = boxed_match.group(1).strip()

    # Look for fraction e.g. 3/4 or -5/2
    frac_match = re.search(r'[-+]?\s*\d+\s*/\s*\d+', s)
    if frac_match:
        try:
            return float(Fraction(frac_match.group(0).replace(' ', '')))
        except Exception:
            pass

    # Look for standard decimal / integer / scientific notation
    # Find all matches and take the last match or explicit answer pattern
    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if matches:
        # Check from last to first for valid float
        for match in reversed(matches):
            try:
                val = float(match)
                if not (val != val):  # not NaN
                    return val
            except ValueError:
                continue

    return None

def verify_numeric(candidate: str, ground_truth: str, tolerance: float = 1e-5) -> bool:
    cand_val = extract_numeric_scalar(candidate)
    gt_val = extract_numeric_scalar(ground_truth)

    if cand_val is None or gt_val is None:
        # Fallback to direct stripped string matching
        cand_str = str(candidate).strip().lower()
        gt_str = str(ground_truth).strip().lower()
        return cand_str == gt_str

    return abs(cand_val - gt_val) <= tolerance

def verify_aime(candidate: str, ground_truth: str) -> bool:
    cand_val = extract_numeric_scalar(candidate)
    gt_val = extract_numeric_scalar(ground_truth)

    if cand_val is None or gt_val is None:
        return str(candidate).strip() == str(ground_truth).strip()

    # AIME answers are exact non-negative integers
    return int(round(cand_val)) == int(round(gt_val))
