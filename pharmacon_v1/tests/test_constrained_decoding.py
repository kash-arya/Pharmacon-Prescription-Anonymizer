"""
Unit tests for Formulary-Constrained Beam Search Trie.
"""

import unittest
from pharmacon_ai.constrained_decoding import FormularyTrie, build_clinical_lexicon


class MockTokenizer:
    """Mock tokenizer mapping words or characters to integer token IDs."""
    def __init__(self):
        self.vocab = {"<eos>": 0, "<pad>": 1, "<unk>": 2}
        self.inv_vocab = {0: "<eos>", 1: "<pad>", 2: "<unk>"}
        self.eos_token_id = 0
        self.pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[word] = new_id
                self.inv_vocab[new_id] = word
            tokens.append(self.vocab[word])
        return tokens


class TestConstrainedDecoding(unittest.TestCase):

    def test_build_clinical_lexicon(self):
        lex = build_clinical_lexicon()
        self.assertGreater(len(lex), 100)
        # Verify clinical phrases exist
        has_paracetamol = any("Paracetamol" in p for p in lex)
        self.assertTrue(has_paracetamol)

    def test_formulary_trie_traversal(self):
        tokenizer = MockTokenizer()
        trie = FormularyTrie()

        # Insert test phrases
        seq1 = tokenizer.encode("Tab Paracetamol 500 mg BD")
        seq2 = tokenizer.encode("Tab Paracetamol 650 mg SOS")
        seq3 = tokenizer.encode("Cap Amoxicillin 500 mg TDS")

        trie.insert(seq1)
        trie.insert(seq2)
        trie.insert(seq3)

        # Test root: next token after empty prefix should be 'Tab' or 'Cap'
        tab_id = tokenizer.vocab["Tab"]
        cap_id = tokenizer.vocab["Cap"]
        allowed_root = trie.get_allowed_tokens([])
        self.assertIn(tab_id, allowed_root)
        self.assertIn(cap_id, allowed_root)

        # Test step after "Tab": next should be "Paracetamol"
        pcm_id = tokenizer.vocab["Paracetamol"]
        allowed_after_tab = trie.get_allowed_tokens([tab_id])
        self.assertEqual(allowed_after_tab, [pcm_id])

        # Test branching after "Tab Paracetamol": should allow "500" and "650"
        id_500 = tokenizer.vocab["500"]
        id_650 = tokenizer.vocab["650"]
        allowed_branch = trie.get_allowed_tokens([tab_id, pcm_id])
        self.assertIn(id_500, allowed_branch)
        self.assertIn(id_650, allowed_branch)


if __name__ == "__main__":
    unittest.main()
