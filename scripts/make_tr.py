import sys
import argparse

def main(args):
    if args.wordlist is None or args.hyphmark is None:
        print("Required arguments missing. Please provide wordlist to translate and hyphenation mark.", file=sys.stderr)
        return
    chars = set()
    left_hyph_min, right_hyph_min = -1, -1
    with open(args.wordlist) as wlh:
        for line in wlh:
            if line.startswith("#"):
                continue
            hyph_indices = []
            line = line.strip()
            for i, c in enumerate(line):
                if c == args.hyphmark:
                    hyph_indices.append(i)
                    continue
                chars.add(c.lower())
            if hyph_indices and (left_hyph_min == -1 or hyph_indices[0] < left_hyph_min):
                left_hyph_min = hyph_indices[0]
            if hyph_indices and (right_hyph_min == -1 or len(line) - hyph_indices[-1] - 1 < right_hyph_min):
                right_hyph_min = len(line) - hyph_indices[-1] - 1

    if args.left_hyphen_min is not None:
        left_hyph_min = args.left_hyphen_min
    if args.right_hyphen_min is not None:
        right_hyph_min = args.right_hyphen_min

    with open(args.wordlist + ".tra", "w") as tra:
        print(f" {left_hyph_min:<2}{right_hyph_min:<2}  {args.hyphmark}", file=tra)
        for char in sorted(chars):
            print(f" {char} {char.upper()}", file=tra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wordlist", help="Path to wordlist, for which the translate file will be created.")
    parser.add_argument("--hyphmark", default="-", required=False, help="Hyphenation mark used in wordlist.")
    parser.add_argument("--left_hyphen_min", required=False, type=int, help="Fixed value for patgen left_hyphen_min hyperparameter")
    parser.add_argument("--right_hyphen_min", required=False, type=int, help="Fixed value for patgen right_hyphen_min hyperparameter")
    args = parser.parse_args()

    main(args)
