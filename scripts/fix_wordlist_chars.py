import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

"""
Fix і→ї and и→й mistakes in a hyphenated Ukrainian wordlist using uk.wiktionary.org.

For each line, the word is stripped of hyphens and queried via the MediaWiki
opensearch API. The first candidate whose letters match the input (with ї
treated as і, й as и, acute accents ignored) is used as the "correct" form.
Characters in the input are replaced in-place: і→ї / и→й only where the
correct form has ї or й at the same letter position. Hyphen positions are
preserved. If no matching candidate is found, the original line is kept.

Output is written incrementally; re-running resumes from the last line
already present in the output file.

Usage:
    python -m scripts.fix_wordlist_chars \\
        --input data/uk/wiktionary/uk-full-wiktionary.wlh \\
        --output results/uk-full-wiktionary_chars_fixed.wlh
"""

API_URL = "https://uk.wiktionary.org/w/api.php"
USER_AGENT = "hyph-bench char-fix script (frantah48@gmail.com)"
COMBINING_ACUTE = "́"


def strip_acute(s: str) -> str:
    return s.replace(COMBINING_ACUTE, "")


def normalize_for_match(s: str) -> str:
    return strip_acute(s).lower().replace("ї", "і").replace("й", "и")


def opensearch(word: str, timeout: float = 15.0) -> list[str]:
    query = urllib.parse.urlencode({
        "action": "opensearch",
        "search": word,
        "limit": 10,
        "namespace": 0,
        "format": "json",
    })
    req = urllib.request.Request(
        f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data[1] if len(data) > 1 else []


def find_correct_form(plain_input: str, candidates: list[str]) -> str | None:
    target = normalize_for_match(plain_input)
    for c in candidates:
        stripped = strip_acute(c)
        if normalize_for_match(stripped) == target:
            return stripped
    return None


def fix_char(input_char: str, correct_char: str) -> str:
    lower_in = input_char.lower()
    lower_correct = correct_char.lower()
    if lower_in == "і" and lower_correct == "ї":
        return "Ї" if input_char.isupper() else "ї"
    if lower_in == "и" and lower_correct == "й":
        return "Й" if input_char.isupper() else "й"
    return input_char


def fix_hyphenated(hyphenated: str, correct_plain: str) -> str:
    out = []
    pos = 0
    for ch in hyphenated:
        if ch == "-":
            out.append(ch)
        else:
            out.append(fix_char(ch, correct_plain[pos]))
            pos += 1
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Fix і→ї and и→й in a hyphenated Ukrainian wordlist using uk.wiktionary.org",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, type=str,
                        help="Hyphenated input wordlist (one word per line)")
    parser.add_argument("--output", required=True, type=str,
                        help="Output path (will be created / appended to for resume)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to sleep between API requests (default: 0.5)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries on network error (default: 3)")
    parser.add_argument("--progress-every", type=int, default=50,
                        help="Print progress every N lines (default: 50)")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    start_at = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            start_at = sum(1 for _ in f)
        if start_at > 0:
            print(f"Resuming: {start_at} lines already in {args.output}", file=sys.stderr)

    with open(args.input, "r", encoding="utf-8") as inp, \
         open(args.output, "a", encoding="utf-8") as out:
        for lineno, raw in enumerate(inp, start=1):
            if lineno <= start_at:
                continue
            hyphenated = raw.rstrip("\n")
            if not hyphenated.strip():
                out.write(raw if raw.endswith("\n") else raw + "\n")
                out.flush()
                continue

            plain = hyphenated.replace("-", "")
            candidates: list[str] = []
            for attempt in range(args.retries):
                try:
                    candidates = opensearch(plain)
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                    wait = 2 ** attempt
                    print(f"line {lineno} '{plain}': {e}; retry in {wait}s", file=sys.stderr)
                    time.sleep(wait)
            else:
                print(f"line {lineno} '{plain}': giving up, keeping original", file=sys.stderr)

            correct = find_correct_form(plain, candidates)
            fixed = fix_hyphenated(hyphenated, correct) if correct else hyphenated

            out.write(fixed + "\n")
            out.flush()

            if lineno % args.progress_every == 0:
                marker = " *" if fixed != hyphenated else ""
                print(f"{lineno}: {hyphenated} -> {fixed}{marker}", file=sys.stderr)

            time.sleep(args.delay)


if __name__ == "__main__":
    main()
