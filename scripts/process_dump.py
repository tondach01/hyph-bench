import argparse
import json
import re


def process_accents(hyph, base_word):
    if base_word is None:
        return hyph
    deaccented = ""
    i_hyph = 0
    i_word = 0
    while i_word < len(base_word) and i_hyph < len(hyph):
        if base_word[i_word] == hyph[i_hyph]:
            deaccented += base_word[i_word]
            i_word += 1
        elif hyph[i_hyph] == "-":
            if word[i_word] == " ":
                i_word += 1
                continue
            deaccented += "-"
        else:
            deaccented += base_word[i_word]
            i_word += 1
        i_hyph += 1
    return deaccented


def process_hyph(data, base_word, has_accents = False):
    translated = set()
    for hyph in data:
        hyph_tr = re.sub("‧", "-", hyph)
        hyph_tr = re.sub("·", "-", hyph_tr)  # different hyphenation marks
        hyph_tr = re.sub("\.", "-", hyph_tr)
        hyph_tr = re.sub(r"\s*-\s*", "-", hyph_tr)
        if has_accents:
            hyph_tr = process_accents(hyph_tr.strip("-"), base_word.strip("-"))
        translated.add(hyph_tr)
    return list(translated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",
                        default="en",
                        choices=["cs", "de", "el", "es", "it", "ms", "nl", "pl", "pt", "ru", "tr", "TBD"],
                        required=True,
                        help="language to which the dump file belongs")
    parser.add_argument("--outfile",
                        default="",
                        required=False,
                        help="file to store the output, if not provided ./data/{lang}/wiktionary}/{lang}_{(en)?wiktionary}_{timestamp}.wlh")
    args = parser.parse_args()

    dump_files = {
        "cs": "cs_wiktionary_251001.jsonl",
        "de": "de_wiktionary_251001.jsonl",
        "el": "el_wiktionary_251001.jsonl",
        "es": "es_wiktionary_251001.jsonl",
        "it": "it_wiktionary_251001.jsonl",
        "ms": "ms_wiktionary_251002.jsonl",
        "nl": "nl_wiktionary_251002.jsonl",
        "pl": "pl_enwiktionary_251001.jsonl",
        "pt": "pt_enwiktionary_251001.jsonl",
        "ru": "ru_wiktionary_251002.jsonl",
        "tr": "tr_wiktionary_251001.jsonl"
    }

    dump_filepath = "./data/wikt_dump/" + dump_files[args.lang]

    if not args.outfile:
        name_long = dump_files[args.lang].split(".")[0]
        outfilename = "./data/" + args.lang + "/wiktionary/" + name_long + ".wlh"
    else:
        outfilename = args.outfile

    accents = False
    if args.lang in ["it"]:
        accents = True

    counter = 0
    ambig_count = 0

    outfile = open(outfilename, "w")  # overwrite all previous content of outfile
    word_buf = set()

    with open(dump_filepath) as file:
        for line in file:

            parsed = json.loads(line)

            word = None
            if "word" in parsed:
                word = parsed["word"]

            hyphenations = list()

            if "hyphenation" in parsed:
                hyphenations = parsed["hyphenation"]
                hyphenations_split = []  # sometimes it is ["hyph1, hyph2"]
                if isinstance(hyphenations, str):
                    hyphenations = [hyphenations]
                for h in hyphenations:
                    h = re.sub(r"\(.*\)\s*|\[.*]:?\s*", "", h)  # sometimes it is "(traditional) hyph" or similar
                    hyphenations_split += h.split(", ")
                hyphenations = hyphenations_split
            elif "hyphenations" in parsed:
                for h in parsed["hyphenations"]:
                    h = re.sub("-+", "-", "-".join(h["parts"]))
                    hyphenations.append(re.sub(r"\(.*\)\s*|\[.*]:?\s*", "", h))

            processed = process_hyph(hyphenations, word, has_accents=accents)
            if(len(processed)) > 1:
                ambig_count += 1

            hyphenations = processed
            processed = []

            for hyphenation in hyphenations:  # process multi-word entries
                processed += hyphenation.split()

            for hyphenation in processed:
                if hyphenation in word_buf:
                    continue
                outfile.write(hyphenation + "\n")
                word_buf.add(hyphenation)
                counter += 1

    outfile.close()

    print(f"Parsed {counter} words from {dump_filepath} into {outfilename}, {ambig_count} with multiple hyphenations.")
