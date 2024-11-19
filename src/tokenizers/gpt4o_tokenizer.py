"""
Extend from the Regex tokenizer
Load the pretrained tokenizer for the gpt4o model & construct the encoding and merges.
"""

import tiktoken
from src.tokenizers.regex_tokenizer import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    # encoding to construct the merge tree
    """
    Given a token, create a byte pair encoding (BPE) of that token.
    Recursively generates merges the split tokens wrt to the min rank in the mergeable_ranks.
    This function helps to recover pair that was merged to get the max_rank.

    Args:
        mergeable_ranks (dict): a dictionary mapping from a pair of byte strings
            to a rank. The rank is the number of times the pair of byte strings
            occurs in the corpus when constructing the merge tree.
        token (bytes): the token to be encoded.
        max_rank (int, optional): the maximum rank of the BPE. If a pair has a
            rank larger than or equal to `max_rank`, we do not merge them. If
            `None`, there is no limit on the rank.

    Returns:
        list: a list of byte strings. Each element in the list is a byte string
            that is a part of the BPE of the token.
    """
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )
    return parts


def recover_merges(mergeable_ranks):
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(
            bpe(mergeable_ranks, token, max_rank=rank)
        )  # the max_rank is the rank of the current merge
        assert len(pair) == 2
        # recover the merge
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges


# this pattern is used for encoding & training
GPT4O_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

GPT4O_SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}


class GPT4OTokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4O_PATTERN)
        # get the encoding (tiktokenizer)
        enc = tiktoken.get_encoding("o200k_base")
        mergeable_ranks = enc._mergeable_ranks
        # recover the merges
        self.merges = recover_merges(mergeable_ranks)
        # reconstruct the vocab
        # NOTE: take into account the byte shuffle from the GPT model here itself
        vocab = {idx: b for b, idx in mergeable_ranks.items()}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # Override the built vocab by the one from the GPT model
        self.vocab = vocab
        self._inverse_vocab = {v: k for k, v in self.vocab.items()}
        # The initial byte tokens are shuffled in the GPT models
        self.shuffled_bytes = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inv_shuffled_bytes = {v: k for k, v in self.shuffled_bytes.items()}
        # register the special tokens
        self.register_special_tokens(GPT4O_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        return super()._encode_chunk(text_bytes)

    def decode(self, ids):
        return super().decode(ids)

    # This tokenizer is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def save(self, file_prefix):
        # save it similar to the base tokenizer
        super().save(file_prefix, file_message="gpt4o:o200k_base")
