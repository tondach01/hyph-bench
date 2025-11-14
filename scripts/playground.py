import unittest

from hyphenator.hyphenator import Hyphenator
from hyphenator.trie import Trie

class HyphenatorTest(unittest.TestCase):
    def test_it_gratis(self):
        hyphenator = Hyphenator("data/it/wiktionary/20251109093112-3.pat")
        self.assertEqual(hyphenator.hyphenate("gra-tis"), "gra-tis")

class TrieTest(unittest.TestCase):
    def test_trie(self):
        trie = Trie()
        trie.populate("data/it/wiktionary/20251109093112-3.pat")
        self.assertEqual(trie.find(".adi"), [(4,'1')])


if __name__ == '__main__':
    unittest.main()
