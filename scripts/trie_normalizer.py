"""Shared trie-normalizer CLI behavior.

The camera-ready experiments use a proportional trie normalizer:
``trie_normalizer = |D|``, the number of lines in the wordlist.
"""

import argparse
import sys
from typing import Optional


def count_wordlist_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def add_trie_normalizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trie-normalizer",
        type=float,
        default=None,
        help=(
            "DANGEROUS fixed trie-size normalizer override. By default the "
            "normalizer is proportional: trie_normalizer = |D|, the wordlist "
            "line count. Fixed normalizers should not usually be used."
        ),
    )
    parser.add_argument(
        "--proportional-normalizer",
        action="store_true",
        help=(
            "Compatibility no-op: proportional trie normalization is now the "
            "default and uses trie_normalizer = |D|."
        ),
    )


def fixed_trie_normalizer_warning(script: str, value: float, phase: str) -> str:
    return "\n".join(
        [
            "",
            "!" * 78,
            f"{phase}: FIXED TRIE SIZE NORMALIZER IN USE in {script}",
            f"fixed trie_normalizer = {value:g}",
            "A fixed trie size normalizer should not usually be used.",
            "A proportional trie normalizer is to be used instead:",
            "    trie_normalizer = |D|  (the wordlist line count)",
            "This run may be inconsistent with camera-ready paper settings.",
            "!" * 78,
            "",
        ]
    )


def warn_fixed_trie_normalizer(script: str, value: float, phase: str) -> None:
    print(fixed_trie_normalizer_warning(script, value, phase), file=sys.stderr)


def resolve_trie_normalizer(
    args: argparse.Namespace,
    wordlist_path: str,
    script: str,
    dataset: Optional[str] = None,
    wordlist_size: Optional[int] = None,
) -> tuple[float, bool]:
    """Return ``(trie_normalizer, uses_fixed_normalizer)``."""
    if args.trie_normalizer is not None:
        value = float(args.trie_normalizer)
        warn_fixed_trie_normalizer(script, value, "START WARNING")
        return value, True

    value = float(
        wordlist_size
        if wordlist_size is not None
        else count_wordlist_lines(wordlist_path)
    )
    label = f" for {dataset}" if dataset else ""
    print(f"Proportional trie_normalizer{label}: |D| = {value:.0f}")
    return value, False
