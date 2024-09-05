import re
from collections import defaultdict
from typing import Dict, Tuple


class BytePairEncoding:
    """
    Implements the Byte Pair Encoding (BPE) algorithm for tokenization.
    
    This class provides methods to learn subword units and tokenize text
    based on the most frequent character pair combinations.
    
    ### Q: What is BPE, and how does one implement it?
    [GeeksForGeeks](https://www.geeksforgeeks.org/byte-pair-encoding-bpe-in-nlp/)
    1. Initialize the vocabulary with all the bytes or characters in the text corpus
    2. Calculate the frequency of each byte or character in the text corpus.
    3. Repeat the following steps until the desired vocabulary size is reached:
        1. Find the most frequent pair of consecutive bytes or characters in the text corpus
        2. Merge the pair to create a new subword unit.
        3. Update the frequency counts of all the bytes or characters that contain the merged pair.
    4. Add the new subword unit to the vocabulary.
    
    The class is heavily inspired from the article, and page 23 of the course book (:
    """

    def __init__(self, corpus: str = None):
        """
        Initializes the BPE model with a corpus.

        Args:
            corpus: The training text used to learn merges.
        """
        self.corpus = corpus
        self._vocab = self._get_initial_vocab() if corpus else {}
        self.merges = []  # Tracks the order of merges

    def _get_initial_vocab(self) -> Dict[str, int]:
        """
        Constructs the initial vocabulary from the corpus.
        
        Returns:
            A dictionary of tokenized words and their frequencies.
        """
        vocab = defaultdict(int)
        for word in self.corpus.split():
            word += "_"  # Appending end-of-word symbol
            vocab[" ".join(list(word))] += 1
        return vocab

    def _get_stats(self) -> Dict[Tuple[str, str], int]:
        """
        Computes the frequency of adjacent symbol pairs in the vocabulary.
        
        Returns:
            A dictionary of symbol pairs and their frequencies.
        """
        pairs = defaultdict(int)
        for word, freq in self._vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str]) -> None:
        """
        Executes the merge operation for the most frequent symbol pair.
        
        Args:
            pair: The symbol pair to be merged.
        """
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word in self._vocab:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = self._vocab[word]

        self._vocab = v_out
        self.merges.append(pair)  # Record the merge

    def train(self, n: int, verbose=0) -> None:
        """
        Performs the BPE algorithm for a specified number of merges.
        
        Args:
            n: Number of merge operations to perform.
            verbose: Controls the level of output during training.
        """
        if verbose == 2:
            print(f"Initial Vocabulary: {self._vocab}\n")

        for i in range(n):
            pairs = self._get_stats()
            if not pairs:
                break  # No more pairs to merge

            max_freq = max(pairs.values())
            best_pairs = [pair for pair in pairs if pairs[pair] == max_freq]
            
            # tie-breaking  logic
            best = min(best_pairs, key=lambda x: (x[1], x[0]))  # Sort by right symbol first, then left

            self._merge_vocab(best)

            if verbose == 1 and (i + 1) % 100 == 0:
                print(f"{i}/{n}", end="\r")
            elif verbose == 2:
                print(f"Step {i}/{n}: Merge {best} -> Vocabulary: {self._vocab}\n")

    def tokenize(self, word: str) -> str:
        """
        Applies learned BPE merges to tokenize a  word.
        
        Args:
            word: The input word to be tokenized.
        
        Returns:
            The tokenized version of the input word.
        """
        tokens = list(word)
        tokens.append("_")  # Append end-of-word symbol

        for merge in self.merges:
            merge_str = "".join(merge)
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens[i:i + 2] = [merge_str]  # Apply merge
                else:
                    i += 1
        
        return " ".join(tokens)
    
    def mass_tokenize(self, words: list) -> list:
        """
        Efficiently tokenizes a list of words using learned BPE merges.
        
        Args:
            words: List of words to tokenize.
            verbosity: Controls the level of progress output.
        
        Returns:
            A list of tokenized words.
        """
        tokens_list = [" ".join(list(word)) + " _" for word in words]
        total_tokens = len(tokens_list)

        for merge in self.merges:
            merge_str = " ".join(merge)
            replacement = "".join(merge)

            for i in range(total_tokens):
                tokens_list[i] = tokens_list[i].replace(merge_str, replacement)

        return [word.replace(" </w>", "") for word in tokens_list]

    def load_merges(self, vocab_path: str) -> None:
        """
        Loads BPE merge operations from a file.
        
        Args:
            vocab_path: Path to the vocab file containing merge operations.
        """
        with open(vocab_path, "r") as f:
            for line in f:
                merge = tuple(line.strip().split())
                self.merges.append(merge)

    @property
    def vocab(self) -> Dict[str, int]:
        """
        Provides access to the current vocabulary after training.
        
        Returns:
            The vocabulary dictionary post-BPE operations.
        """
        return self._vocab


if __name__ == "__main__":
    import time
    start_time = time.time()

    import argparse

    # Command-line interface setup ### straight from this tutorial: https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(description="BPE Tokenizer for subword segmentation")

    ### learn bpe
    parser.add_argument("--learn_bpe", action="store_true", help="Flag to initiate BPE learning")
    parser.add_argument("--inpath", help="Path to the input text file")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Target vocabulary size")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for output")
    parser.add_argument("--outpath", type=str, required=True, help="Path for saving tokenized output")
    parser.add_argument("--vocab", type=str, help="Path for saving or loading learned merge operations")

    ### apply bpe
    parser.add_argument("--apply_bpe", action="store_true", help="Flag to apply BPE merges from vocab file")

    args = parser.parse_args()

    if args.learn_bpe:
        with open(args.inpath, "r") as file:
            corpus = file.read()

        k = args.vocab_size

        # Initialize and train BPE model
        bpe = BytePairEncoding(corpus)
        bpe.train(k, verbose=args.verbose)
        
        if args.verbose == 2:
            print(f"Final Vocabulary: {bpe.vocab}")

        # save the tokenized input
        if args.outpath:
            tokenized_lines = bpe.mass_tokenize(corpus.split())
    
            with open(args.outpath, "w") as outfile:
                outfile.write("\n".join(tokenized_lines) + "\n")
    
        # Save learned merge operations
        if args.vocab:
            with open(args.vocab, "w") as vocabfile:
                for merge in bpe.merges:
                    vocabfile.write(f"{merge[0]} {merge[1]}\n")

    elif args.apply_bpe:
        # Load corpus and vocab
        with open(args.inpath, "r") as infile:
            corpus = infile.read()

        bpe = BytePairEncoding()
        bpe.load_merges(args.vocab)

        tokenized_lines = bpe.mass_tokenize(corpus.split())
    
        # Save tokenized output
        with open(args.outpath, "w") as outfile:
            outfile.write("\n".join(tokenized_lines) + "\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
