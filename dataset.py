import random

import torch
from torch.utils.data import Dataset


class BracketDataset(Dataset):
    def __init__(self, num_samples=10000, max_len=10):
        self.BRACKET_MAP = {"(": ")", "[": "]", "{": "}", "<": ">", "¿": "?"}
        self.VOCAB = {
            ch: i
            for i, ch in enumerate(
                self.BRACKET_MAP.keys() | self.BRACKET_MAP.values(), start=1
            )
        }
        self.VOCAB["PAD"] = 0  # Padding token
        self.VOCAB["<EOS>"] = max(self.VOCAB.values()) + 1
        self.VOCAB["<SOS>"] = max(self.VOCAB.values()) + 1

        self.ID2VOCAB = {v: k for k, v in self.VOCAB.items()}
        self.max_len = max_len + 2

        self.data = []
        for _ in range(num_samples):
            question, answer = self.generate_bracket_data(max_len)
            input_tokens = self.tokenize(question)
            output_tokens = self.tokenize(answer)
            self.data.append((input_tokens, output_tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tokens, output_tokens = self.data[idx]
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(
            output_tokens, dtype=torch.long
        )

    def tokenize(self, expression):
        """Tokenize a string expression into a list of token IDs."""
        return [self.VOCAB[ch] for ch in expression.split() if ch in self.VOCAB]

    def decode(self, tokens):
        """Convert a tensor or list of token IDs back into a string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return " ".join(
            self.ID2VOCAB[token]
            for token in tokens
            # if token in self.ID2VOCAB and self.ID2VOCAB[token] not in ["PAD", "<EOS>"]
        )

    def generate_bracket_data(self, max_depth=5, padd=True):
        """Generate a balanced bracket sequence with a matching closing sequence."""
        stack = []
        input_seq = []

        for _ in range(max_depth):
            opening = random.choice(list(self.BRACKET_MAP.keys()))
            stack.append(opening)
            input_seq.append(opening)

        output_seq = [self.BRACKET_MAP[c] for c in reversed(stack)]

        output_seq = self.add_sos(output_seq)
        output_seq = self.add_eos(output_seq)

        input_seq = self.add_sos(input_seq)
        input_seq = self.add_eos(input_seq)

        if padd:
            output_seq = self.add_padding(output_seq)
            input_seq = self.add_padding(input_seq)

        return " ".join(input_seq), " ".join(output_seq)

    def add_padding_tokens(self, tokens):
        return tokens + [self.VOCAB["PAD"]] * (self.max_len - len(tokens))

    def add_padding(self, data):
        return data + ["PAD"] * (self.max_len - len(data))

    def add_sos_token(self, tokens):
        return [self.VOCAB["<SOS>"]] + tokens

    def add_eos_token(self, tokens):
        return tokens + [self.VOCAB["<EOS>"]]

    def add_sos(self, tokens):
        return ["<SOS>"] + tokens

    def add_eos(self, tokens):
        return tokens + ["<EOS>"]
