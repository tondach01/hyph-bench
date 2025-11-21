import argparse
import json
import re


ALLOWED_HYPHENATORS = "‧·.‐­-"
DEACCENTED = {
    "à": "a", "á": "a",
    "ù": "u", "ú": "u",
    "ì": "i", "í": "i",
    "è": "e", "é": "e",
    "ò": "o", "ó": "o",
    "ѝ": "и", "ѐ": "е",
    "ý": "у"
}
ACCENTED = {
    "a": "àá",
    "e": "èé",
    "i": "ìí",
    "o": "òó",
    "u": "ùú",
    "и": "ѝ",
    "е": "ѐ",
    "у": "ý"
}

def process_accents(hyph, base_word):
    if base_word is None:
        return hyph
    deaccented = ""
    i_hyph = 0
    i_word = 0
    while i_word < len(base_word) and i_hyph < len(hyph):
        if base_word[i_word] == hyph[i_hyph] or base_word[i_word] == DEACCENTED.get(hyph[i_hyph], ""):
            deaccented += base_word[i_word]
            i_word += 1
        elif hyph[i_hyph] == "-":
            if word[i_word] == " ":
                i_word += 1
                continue
            deaccented += "-"
        i_hyph += 1
    return deaccented

def build_regex(base_word: str, has_accents: bool = False):
    letters = []
    for letter in base_word:
        if letter in "()[]":
            letters.append(f"[\\{letter}]")
        elif has_accents:
            letters.append(f"[{letter}{ACCENTED.get(letter, '')}]")
        else:
            letters.append(f"[{letter}]")
    return (f"[-]?".join([letter for letter in letters]))+"\s"

def process_hyph(data: list, base_word: str, has_accents: bool = False):
    translated = set()
    for hyph in data:
        hyph_tr = re.sub(r"\d+|\*|\.", "",
                         hyph)  # numbers are reserved for patgen levels, * and . for hyphenation marking
        hyph_tr = re.sub(r"-+", "-", hyph_tr)
        hyph_tr = re.sub(r"\s*-\s*", "-", hyph_tr)
        hyph_tr = hyph_tr.strip(" -")
        if has_accents:
            hyph_tr = process_accents(hyph_tr, base_word.strip("-"))
        translated.add(hyph_tr)
    return list(translated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",
                        default="en",
                        choices=["cs", "de", "el", "es", "it", "ms", "nl", "pl", "pt", "ru", "tr", "TBD"],
                        required=True,
                        help="Language to which the dump file belongs")
    parser.add_argument("--outfile",
                        default="",
                        required=False,
                        help="File to store the output, if not provided ./data/{lang}/wiktionary}/{lang}_{(en)?wiktionary}_{timestamp}.wlh")
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

    dump_filepath = "./wikt_dump/" + dump_files[args.lang]

    if not args.outfile:
        name_long = dump_files[args.lang].split(".")[0]
        outfilename = "./data/" + args.lang + "/wiktionary/" + name_long + ".wlh"
    else:
        outfilename = args.outfile

    accents = False
    if args.lang in ["it", "ru"]:
        accents = True

    counter = 0

    outfile = open(outfilename, "w")  # overwrite all previous content of outfile
    word_buf = set()

    with open(dump_filepath) as file:
        for line in file:

            parsed = json.loads(line)

            word = None
            if "word" in parsed:
                word = parsed["word"].lower()

            hyphenations = ""

            if "hyphenation" in parsed:
                hyphenations = parsed["hyphenation"]
                if isinstance(hyphenations, list):
                    hyphenations = " ".join(hyphenations)
                hyphenations = hyphenations.lower() + " "  # add final space regex matching
            elif "hyphenations" in parsed:
                hyphenations = " ".join(["-".join(h["parts"]) for h in parsed["hyphenations"]]).lower() + " "
            else:
                continue

            if args.lang == "ru":
                hyphenations = re.sub("[•·̀́]", "", hyphenations)
                hyphenations = re.sub("à", "а", hyphenations)
                hyphenations = re.sub("ó", "о", hyphenations)
                hyphenations = re.sub("é", "е", hyphenations)
                hyphenations = re.sub("á", "а", hyphenations)
            hyphenations = re.sub(f"[{ALLOWED_HYPHENATORS}]", "-", hyphenations)  # different hyphenation marks
            regex = build_regex(word, accents)
            candidates = re.findall(regex, hyphenations)
            processed = process_hyph(candidates, word, has_accents=accents)
            if hyphenations and not candidates:
                print(word, hyphenations, candidates)

            hyphenations = []
            for p in processed:
                hyphenations += p.split() # process multi-word entries

            for hyphenation in hyphenations:
                if args.lang == "ru" and "-" not in hyphenation:
                    continue
                if hyphenation in word_buf:
                    continue
                outfile.write(hyphenation + "\n")
                word_buf.add(hyphenation)
                counter += 1

    outfile.close()

    print(f"Parsed {counter} words from {dump_filepath} into {outfilename}.")
