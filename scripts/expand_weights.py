import argparse
import re

def expand_line(line: str, out):
    """
    Replicate line according to its weight.
    :param line: single input line
    :param out: open (for writing) output file
    """
    parsed = re.match("(?P<weight>\d+)(?P<word>\S+)", line)
    if parsed is None:
        return
    for _ in range(int(parsed["weight"])):
        out.write(parsed["word"] + "\n")

def expand_wordlist(wl_file: str, out_file: str = ""):
    """
    Expand weighted wordlist.
    :param wl_file: path to wordlist file
    :param out_file: path to output wordlist file
    """
    if not out_file:
        out_file = wl_file + "_expanded.wlh"
    out = open(out_file, "w")
    with open(wl_file) as wordlist:
        for line in wordlist:
            expand_line(line, out)
    out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to weighted wordlist file.")
    parser.add_argument("--outfile", required=False, default="", type=str, help="Path to output wordlist")
    args = parser.parse_args()

    expand_wordlist(args.file, args.outfile)
