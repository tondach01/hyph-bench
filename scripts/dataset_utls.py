import os
from typing import Tuple, List

# Default pattern ranges per level (from base.in profile)
DEFAULT_PAT_RANGES = [
    (1, 4),   # Level 1
    (2, 5),   # Level 2
    (2, 6),   # Level 3
    (2, 7),   # Level 4
]


def find_dataset(lang: str, data_dir: str = None) -> Tuple[str, str]:
    """
    Find wordlist and translate file for a language.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', lang)

    # Check wiktionary subdirectory first (preferred)
    wikt_dir = os.path.join(data_dir, 'wiktionary')
    if os.path.exists(wikt_dir):
        for f in sorted(os.listdir(wikt_dir)):
            if f.endswith('.wlh'):
                wl = os.path.join(wikt_dir, f)
                tr = wl + '.tra'
                if os.path.exists(tr):
                    return os.path.abspath(wl), os.path.abspath(tr)

    # Check data directory directly
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.wlh'):
                wl = os.path.join(data_dir, f)
                tr = wl + '.tra'
                if os.path.exists(tr):
                    return os.path.abspath(wl), os.path.abspath(tr)

    # Recursive fallback: Find all valid pairs
    candidates = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in sorted(files):
                if f.endswith('.wlh'):
                    wl = os.path.join(root, f)
                    tr = wl + '.tra'
                    if os.path.exists(tr):
                        candidates.append((os.path.abspath(wl), os.path.abspath(tr)))

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        msg = f"Multiple datasets found for {lang}. Please specify --wordlist and --translate explicitly.\nFound:\n"
        for wl, _ in candidates:
            msg += f"  - {wl}\n"
        raise ValueError(msg)

    raise FileNotFoundError(f"No dataset found for language: {lang} in {data_dir}")


def parse_profile(profile_path: str) -> List[Tuple[int, int]]:
    """
    Parse a profile file to get pat_start/pat_finish per level.
    """
    pat_ranges = []
    with open(profile_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pat_ranges.append((int(parts[0]), int(parts[1])))
    return pat_ranges
