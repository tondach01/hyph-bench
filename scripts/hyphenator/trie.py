class TrieNode:
    """
    Single node of a trie structure
    """
    def __init__(self):
        self.output = None
        self.children = dict()

    def put(self, letter: str = "", output = None):
        """
        Add a letter or output to the node
        :param letter: string to be inserted as child of the node
        :param output: data to be stored in the node
        :return: True if no errors occurred, False otherwise
        """
        if letter and letter in self.children.keys():
            return True
        end = not letter
        if output is not None and end:
            if self.output is not None:
                print("Output was already set")
                return False
            self.output = output
        elif not end:
            self.children[letter] = TrieNode()
        return True


class Trie:
    """
    Trie structure for dictionary storing
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, outputs: str = "123456789"):
        """
        Insert a word into trie
        :param word: string to be inserted
        :param outputs: symbols in the string to be treated as outputs
        :return: True if word was inserted successfully, False otherwise
        """
        current = self.root
        output = []
        n_outputs = 0
        for i,letter in enumerate(word):
            if letter in outputs:
                output.append((i-n_outputs, letter))
                n_outputs += 1
            elif not current.put(letter, output):
                return False
            else:
                current = current.children[letter]
        return current.put(output=output)

    def find(self, word: str, ignore: str = "-"):
        """
        Find trie entry for given word
        :param word: string to be searched
        :param ignore: symbols in the string to be ignored
        :return: output of corresponding trie node if present, else None
        """
        current = self.root
        for letter in word:
            if letter in ignore:
                continue
            if letter not in current.children.keys():
                return None
            current = current.children[letter]
        return current.output

    def populate(self, wordlist: str, outputs: str = "12345678"):
        """
        Insert all words from a wordlist (one per line) into trie
        :param wordlist: path to possibly hyphenated wordlist
        :param outputs: all symbols representing output
        """
        with open(wordlist) as wl:
            for line in wl:
                if not self.insert(line.strip(), outputs=outputs):
                    break
