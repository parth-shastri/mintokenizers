from abc import ABC, abstractmethod
from src.tokenizers.utils import render_token


# Tokenizer base class
class Tokenizer(ABC):
    def __init__(self, pattern=None):
        # default: vocab 256
        self.merges = {}  # (int, int) -> (int)
        self.pattern = pattern or ""
        self.special_tokens = {}  # str -> int,
        self.vocab, self._inverse_vocab = self._build_vocab()  # int -> bytes

    @abstractmethod
    def train(self, text, vocab_size, verbose=False):
        # Given the text and the vocab size train the BPE tokenizer
        # i.e. build a bottom-up tree by merging tokens and adding to the vocab
        pass

    @abstractmethod
    def encode(self, text):
        # given the text encode it to ids using the trained tokenizer
        pass

    @abstractmethod
    def decode(self, ids):
        # given the integer ids, decode them to a str.
        pass

    def _build_vocab(self, base_vocab=None):
        # vocab is determined from the merges or the base_vocab
        # In case of the openai GPT models, the base vocab is shuffled
        vocab = {idx: bytes([idx]) for idx in range(256)} if base_vocab is None else base_vocab
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        # build the inverse vocab (will be used for the encoding)
        inv_vocab = {v: k for k, v in vocab.items()}
        return vocab, inv_vocab

    def save(self, file_prefix, file_message=None):
        # .model and .vocab file
        # write the model
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            # write the version, pattern and merges
            file_message = "bpe-minimal" if file_message is None else file_message
            f.write(f"{file_message} v1\n")
            f.write(f"{self.pattern}\n")

            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # write the vocab
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens can be partial utf-8 sequence
                #       therefore can't be decoded into valid strings
                #       using errors='replace'
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # this is a leaf token with no merged token
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file, file_message=None):
        "Inverse of the save() method only with the .model file"
        file_message = "bpe-minimal" if file_message is None else file_message
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == f"{file_message} v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # num special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

            self.merges = merges
            self.special_tokens = special_tokens
            self.vocab, _ = self._build_vocab()
