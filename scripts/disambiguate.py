import re
import argparse


WORD_LEN_LIMIT = 50


def hyph_indices(word: str, hyphenation_mark: str = "-"):
    """
    Find indices of hyphenation points within given word
    :param word: string to be searched
    :param hyphenation_mark: string used as hyphenation mark
    """
    return set([match.start() for match in re.finditer(hyphenation_mark, word)])


def superset(first: set, second: set):
    """
    Check whether there is a superset-subset relation between two sets
    :param first: set to be checked
    :param second: set to be checked
    :return: 0 if relation is present and first is the superset 1 if second is the superset, -1 otherwise
    """
    if first.issubset(second):
        return 1
    elif second.issubset(first):
        return 0
    return -1


def disambiguate(file: str, hyphenation_mark: str = "-", outfile: str = ""):
    """
    Check and resolve inconsistencies in given dataset by joining or keeping words with non-unique hyphenations
    :param file: path to word list to be processed
    :param hyphenation_mark: string used as hyphenation mark
    :param outfile: path to output file
    :return: number of ambiguities (before, after) processing
    """
    words = dict()
    found, disambiguated = 0, 0
    with open(file) as wl:
        for new in wl:
            new = new.strip()
            word = re.sub(hyphenation_mark, "", new)
            if word not in words:
                words[word] = [new]
                continue
            found += 1
            new_ind = hyph_indices(new, hyphenation_mark)
            resolved = False
            for i, hyphenation in enumerate(words[word]):
                hyph_ind = hyph_indices(hyphenation, hyphenation_mark)
                superset_index = superset(new_ind, hyph_ind)
                if superset_index != -1:
                    if superset_index == 0:
                        words[word][i] = new
                    disambiguated += 1
                    resolved = True
                    break
            if not resolved:
                words[word].append(new)

    if not outfile:
        outfile = file+"_dis.wlh"

    with open(outfile, "w") as out:
        for word in sorted(words.keys()):
            for hyphenation in sorted(words[word]):
                if len(hyphenation) > WORD_LEN_LIMIT:
                    continue
                out.write(hyphenation + "\n")

    return found, found - disambiguated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="File to be disambiguated")
    parser.add_argument("-m", "--hyphmark", type=str, required=False, default="-", help="String used as hyphenation mark")
    parser.add_argument("-o", "--outfile", type=str, required=False, default="", help="File to write output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose printout")
    args = parser.parse_args()

    before, after = disambiguate(args.file, hyphenation_mark=args.hyphmark, outfile=args.outfile)
    if args.verbose:
        print(f"Disambiguated {args.file} into {args.outfile if args.outfile else args.file+'_dis.wlh'}, ambiguous hyphenations reduced from {before} to {after}")
    else:
        if "\\" in args.file:
            split = args.file.split("\\")
        else:
            split = args.file.split("/")
        print("" if len(split) < 3 else split[-3], "" if len(split) < 2 else split[-2], before, after)
