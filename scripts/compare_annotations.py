import sys
import os

def select_divs(word: str) -> str:
    result = ""
    i = 1

    while i < len(word):
        if word[i] != "-":
            result += " "
            i += 1
        else:
            result += "-"
            i += 2

    return result

def get_word(line: str) -> str:
    return line.strip().split("=")[1]

def calculate_kappa(wordlist_a: list[str], wordlist_b: list[str]):
    tt, ff, tf, ft = 0, 0, 0, 0

    for i in range(min(len(wordlist_a), len(wordlist_b))):
        word_a, word_b = wordlist_a[i], wordlist_b[i]
        
        for i in range(len(word_a)):
            if i == len(word_b):
                break
            if word_b[i] == "-" and word_a[i] == "-":
                tt += 1
            elif word_b[i] == " " and word_a[i] == " ":
                ff += 1
            elif word_b[i] == "-" and word_a[i] == " ":
                tf += 1
            elif word_b[i] == " " and word_a[i] == "-":
                ft += 1


    total = tt + ff + tf + ft
    p_hyphenate = ((tt + tf) / total) * ((tt + ft) / total)
    p_dont_hyphenate = ((ff + tf) / total) * ((ff + ft) / total)
    p_o = (tt + ff) / total
    p_e = p_hyphenate + p_dont_hyphenate
    kappa = (p_o - p_e) / (1 - p_e)

    return kappa

def main():
    if (len(sys.argv) < 3):
        print("At least two files needed")
        return

    files = {}

    for i in range(1, len(sys.argv)):
        words = []
        name = os.path.basename(sys.argv[i])
        
        with open(sys.argv[i], encoding="utf-8") as file:
            for line in iter(lambda: file.readline(), ""):
                word = get_word(line)

                words.append(select_divs(word))

        files[name] = words

    col_spec = "l|" + "X" * len(files)
    names = list(files.keys())

    print(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    print(f"  & {' & '.join(names)} \\\\")
    print("  \\hline")

    for name_a, words_a in files.items():
        results = []
        for name_b, words_b in files.items():
            kappa = calculate_kappa(words_a, words_b) if name_a != name_b else 1.0
            results.append(f"{kappa:.4f}")
        print(f"  {name_a} & {' & '.join(results)} \\\\")

    print("\\end{tabularx}")

if __name__ == "__main__":
    main()
    