"""Deterministic, surface-form-disjoint dataset splitting."""

import hashlib
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple


_WEIGHTED_LINE = re.compile(r"(?P<weight>\d+)(?P<word>\S+)")
_SPLIT_NAMES = ("train", "validation", "test")
WordEntry = Tuple[str, str, int]


def _canonical_word(annotation: str) -> str:
    return unicodedata.normalize("NFC", annotation.replace("-", "").casefold())


def _weighted_source(wordlist_path: str) -> str | None:
    if wordlist_path.endswith(".wlhw"):
        return wordlist_path
    expanded_suffix = "_expanded.wlh"
    if wordlist_path.endswith(expanded_suffix):
        candidate = wordlist_path[: -len(expanded_suffix)]
        if os.path.isfile(candidate) and candidate.endswith(".wlhw"):
            return candidate
    return None


def resolve_word_entries(wordlist_path: str) -> Tuple[List[WordEntry], bool, str]:
    """Resolve duplicate surface forms and their source priorities."""
    source_path = _weighted_source(wordlist_path)
    input_path = source_path or wordlist_path
    weighted = source_path is not None

    # canonical form -> annotation -> [source priority, first source line]
    groups: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
    with open(input_path, encoding="utf-8") as wordlist:
        for line_number, raw_line in enumerate(wordlist, 1):
            text = raw_line.strip()
            if weighted:
                match = _WEIGHTED_LINE.fullmatch(text)
                if match is None:
                    raise ValueError(
                        f"{input_path}:{line_number}: malformed weighted entry"
                    )
                priority = int(match.group("weight"))
                annotation = match.group("word")
                if priority < 1:
                    raise ValueError(
                        f"{input_path}:{line_number}: weight must be positive"
                    )
            else:
                if not text:
                    raise ValueError(f"{input_path}:{line_number}: empty word-list entry")
                priority = 1
                annotation = text

            canonical = _canonical_word(annotation)
            variants = groups[canonical]
            if annotation in variants:
                variants[annotation][0] = max(variants[annotation][0], priority)
            else:
                variants[annotation] = [priority, line_number]

    entries: List[WordEntry] = []
    for canonical, variants in groups.items():
        annotation, (priority, _) = min(
            variants.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
        entries.append((canonical, annotation, priority))
    return entries, weighted, source_path or ""


def rank_word_entries(entries: List[WordEntry], seed: int = 42) -> None:
    """Sort entries in place by a stable seeded content digest."""
    seed_bytes = seed.to_bytes(8, byteorder="big", signed=True)
    entries.sort(
        key=lambda entry: (
            hashlib.sha256(seed_bytes + b"\0" + entry[0].encode("utf-8")).digest(),
            entry[0],
        )
    )


def create_clean_split(
    wordlist_path: str, output_dir: str, seed: int = 42
) -> Dict[str, str]:
    """Create grouped 8/1/1 splits, expanding priorities in training only."""
    os.makedirs(output_dir, exist_ok=True)
    entries, weighted, source_path = resolve_word_entries(wordlist_path)
    rank_word_entries(entries, seed)

    train_end = len(entries) * 8 // 10
    validation_end = train_end + len(entries) // 10
    partitions = {
        "train": entries[:train_end],
        "validation": entries[train_end:validation_end],
        "test": entries[validation_end:],
    }

    paths = {
        name: os.path.abspath(os.path.join(output_dir, f"data.{name}.wlh"))
        for name in _SPLIT_NAMES
    }
    paths["unique"] = os.path.abspath(os.path.join(output_dir, "data.unique.wlh"))
    line_counts = {}
    type_counts = {}
    for name, partition in partitions.items():
        type_counts[name] = len(partition)
        line_counts[name] = 0
        with open(paths[name], "w", encoding="utf-8") as handle:
            for _, annotation, priority in partition:
                repetitions = priority if weighted and name == "train" else 1
                handle.write((annotation + "\n") * repetitions)
                line_counts[name] += repetitions

    with open(paths["unique"], "w", encoding="utf-8") as handle:
        for _, annotation, _ in entries:
            handle.write(annotation + "\n")

    return {
        **paths,
        **{f"{name}_count": str(line_counts[name]) for name in _SPLIT_NAMES},
        **{f"{name}_type_count": str(type_counts[name]) for name in _SPLIT_NAMES},
        "unique_count": str(len(entries)),
        "split_seed": str(seed),
        "split_method": "sha256_grouped_8_1_1",
        "weighted_training": str(weighted).lower(),
        "weighted_source": os.path.abspath(source_path) if source_path else "",
    }
