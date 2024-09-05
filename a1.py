import re
from collections import defaultdict
from typing import Dict, Tuple


class BytePairEncoding:
    """
    Implements the Byte Pair Encoding (BPE) algorithm for tokenizing text by iteratively merging the most frequent pairs of symbols.
    """

    def __init__(self, corpus: str):
        """
        Initializes the BPE model with a corpus.

        :param corpus: A string containing the training text for BPE.
        """
        self.corpus = corpus
        self._vocab = self._get_initial_vocab()
        self.merges = []  # To store the order of merges

    def _get_initial_vocab(self) -> Dict[str, int]:
        """
        Builds the initial vocabulary from the corpus. Each word in the corpus is split into characters, and each character
        sequence is considered as an initial token.

        :return: A dictionary where keys are tokenized words and values are their frequencies in the corpus.
        """
        vocab = defaultdict(int)
        for word in self.corpus.split():
            word += "_"
            # Each character in the word is separated by a space to form the initial vocabulary
            vocab[" ".join(list(word))] += 1
        return vocab

    def _get_stats(self) -> Dict[Tuple[str, str], int]:
        """
        Calculates the frequency of each pair of consecutive symbols in the vocabulary.

        :return: A dictionary where keys are pairs of symbols and values are their frequencies.
        """
        pairs = defaultdict(int)
        for word, freq in self._vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                # Count the frequency of each adjacent pair of symbols
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str]) -> None:
        """
        Merges the most frequent pair of symbols in the vocabulary.

        :param pair: A tuple representing the pair of symbols to be merged.
        """
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word in self._vocab:
            # Replace the most frequent pair with its merged form in the vocabulary
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = self._vocab[word]

        self._vocab = v_out
        self.merges.append(pair)  # Store the merge operation

    ### this is the token learner. TODO: Change to look more like the books pseudo code on page 23 ###
    def train(self, n: int, verbose=0) -> None:
        """
        Performs the BPE algorithm for a specified number of merges.

        :param n: The number of merge operations to perform.
        """
        print(f"Initial Vocabulary: {self._vocab}\n")

        for i in range(n):
            pairs = self._get_stats()
            if not pairs:
                break  # No more pairs to merge

            # Find the most frequent pair and handle ties
            max_freq = max(pairs.values())
            best_pairs = [pair for pair in pairs if pairs[pair] == max_freq]

            # Tie-breaking: choose the pair that results in the smallest merged token
            best = min(best_pairs, key=lambda x: len("".join(x)))

            self._merge_vocab(best)

            if verbose == 1:
                print(f"{i}/{n}", end="\r")
            elif verbose == 2:
                print(f"Step {i}/{n}: Merge {best} -> Vocabulary: {self._vocab}\n")

    def tokenize(self, word: str) -> str:
        """
        Tokenizes a new word using the learned BPE merges.

        :param word: The word to tokenize.
        :return: The tokenized version of the word.
        """
        word = (
            " ".join(list(word)) + " </w>"
        ) + "_"  # Initialize the word with spaces between characters # _ to represent end of word
        for merge in self.merges:
            bigram = re.escape(" ".join(merge))  # Escape the bigram for regex safety
            word = re.sub(r"\b" + bigram + r"\b", "".join(merge), word)
        return word.replace("</w>", "")

    @property
    def vocab(self) -> Dict[str, int]:
        """
        Returns the current vocabulary after training.

        This property ensures that the vocabulary is accessed in a read-only manner.

        :return: The vocabulary dictionary after BPE merges.
        """
        return self._vocab


if __name__ == "__main__":
    import argparse

    ### straight from this tutorial: https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(description="BPE Tokenizer")

    parser.add_argument(
        "--learn_bpe", action="store_true", help="Flag to run BPE learning"
    )
    parser.add_argument(
		"--inpath", help="Path to input text"
	)
    parser.add_argument(
		"--vocab_size",
  		type=int,
		default=10_000,
		help="size of the vocabulary"
	)
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Set verbosity level: 0 (no output), 1 (minimal output), 2 (detailed output)",
    )
    parser.add_argument(
    	"--outpath", type=str, required=True, help="Path to save the tokenized output text"
	)
    parser.add_argument(
        "--vocab", type=str, required=True, help="Path to save the list of merge operations"
    )

    args = parser.parse_args()

    if args.learn_bpe:
        with open(args.inpath, "r") as file:
            corpus = file.read()

        # k = 10_000
        k = args.vocab_size

        # Initialize and train the BPE model
        bpe = BytePairEncoding(corpus)
        bpe.train(k, verbose=args.verbose)

        # Print the final vocabulary after training
        final_vocab = bpe.vocab
        print(f"Final Vocabulary: {final_vocab}")
        
        if args.outpath:
            with open(args.outpath, "w") as outfile:
                word_count = 0
                total_words = len(corpus.split())
                for word in corpus.split():
                    word_count += 1
                    tokenized_word = bpe.tokenize(word)
                    if args.verbose == 1:
                        print(tokenized_word)
                    elif args.verbose == 2:
                        print(f"Original Word: {word}, Tokenized Word: {tokenized_word}")
                    print(f"Word {word_count}/{total_words}")
                    outfile.write(tokenized_word + "\n")
                    
        # Write the learned merge operations to the vocab file
        if args.vocab:
            with open(args.vocab, "w") as vocabfile:
                for i, merge in enumerate(bpe.merges):
                    if args.verbose == 1:
                        print(f"{i}/{len(bpe.merges)}")
                    if args.verbose == 2:
                        print(f"{i}/{len(bpe.merges)} Writing merge: {merge[0]} {merge[1]}")
                    vocabfile.write(f"{merge[0]} {merge[1]}\n")
