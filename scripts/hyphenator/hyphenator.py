import trie

class Hyphenator:
    """
    Simple word hyphenator
    """
    def __init__(self, patterns: str = "", word_boundary: str = ".", hyphenation_mark: str = "-", left_hyphen_min: int = 1, right_hyphen_min: int = 1):
        self.patterns = trie.Trie()
        if patterns:
            self.patterns.populate(patterns)
        self.word_boundary = word_boundary if word_boundary else "."
        self.hyphenation_mark = hyphenation_mark
        self.left_hyphen_min = left_hyphen_min
        self.right_hyphen_min = right_hyphen_min

    def hyphenate(self, word: str):
        """
        Hyphenate a word using stored patterns
        :param word: string to be hyphenated
        :return: word with injected hyphenation marks
        """
        word_bounded = self.word_boundary + word.lower() + self.word_boundary
        levels = [0 for _ in range(len(word_bounded)-1)]

        for i in range(len(word_bounded)-1):
            for j in range(i+1, len(word_bounded) + 1):
                subword = word_bounded[i:] if j == len(word_bounded) else word_bounded[i:j]
                outputs = self.patterns.find(subword, self.hyphenation_mark)
                if outputs is None:
                    continue
                for index, value in outputs:
                    levels[i + index - 1] = max(int(value), levels[i + index - 1])
        hyphenated = ""
        for i,letter in enumerate(word):
            if levels[i] > 0 and levels[i] % 2 == 1 and self.left_hyphen_min <= i <= len(word) - self.right_hyphen_min:
                hyphenated += self.hyphenation_mark
            hyphenated += letter
        return hyphenated

if __name__ == "__main__":
    hyphenator = Hyphenator("../../data/it/wiktionary/20251109093112-3.pat")
    print(hyphenator.hyphenate("auto"))
