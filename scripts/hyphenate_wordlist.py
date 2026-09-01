"""Apply a generated PATGEN pattern file to a plain word list."""

import argparse
from pathlib import Path

from .hyphenator.hyphenator import Hyphenator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyphenate one word per line with a PATGEN pattern file."
    )
    parser.add_argument("--wordlist", required=True, type=Path)
    parser.add_argument("--patterns", required=True, type=Path)
    parser.add_argument("--translate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--separator",
        default="-",
        help="Hyphenation mark to write (default: -).",
    )
    args = parser.parse_args()

    output_path = args.output or args.wordlist.with_suffix(
        args.wordlist.suffix + ".hyph"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hyphenator = Hyphenator(
        str(args.patterns),
        hyphenation_mark=args.separator,
        translate_file=str(args.translate),
    )

    count = 0
    with (
        args.wordlist.open(encoding="utf-8") as wordlist,
        output_path.open("w", encoding="utf-8") as output,
    ):
        for line in wordlist:
            word = line.strip()
            if not word:
                continue
            output.write(hyphenator.hyphenate(word) + "\n")
            count += 1

    print(f"Hyphenated {count} words into {output_path}")


if __name__ == "__main__":
    main()