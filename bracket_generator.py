import json
from typing import List

import torch
import typer

from model import TinyTransformer, load_model

app = typer.Typer()


model = TinyTransformer(
    vocab_size=13,
    d_model=16,
    num_heads=4,
    num_layers=4,
    ff_dim=128,
    max_len=12,
)


model = load_model(model, "model.pth", device="cuda")

with open("vocab.json") as f:
    tokenizer = json.load(f)


reverse_tokenizer = {v: k for k, v in tokenizer.items()}


def text_to_tokens(text: str) -> List[int]:
    return [tokenizer.get(char, 0) for char in text.split()]


def tokens_to_text(tokens: List[int]) -> str:
    return " ".join([reverse_tokenizer.get(token, "<UNK>") for token in tokens])


@app.command()
def generate(input_text: str, max_len: int = 20):
    # Step 1: Tokenize input text
    input_tokens = text_to_tokens(
        "<SOS> " + input_text + " <EOS>"
    )  # Add <SOS> and <EOS> tokens
    input_tensor = torch.tensor(input_tokens).to("cuda")  # Add batch dimension

    # Step 2: Generate output using the model
    output_tokens = model.generate(
        input_tensor,
        eos_token=tokenizer["<EOS>"],
        sos_token=tokenizer["<SOS>"],
        max_len=max_len,
    )

    output_text = tokens_to_text(output_tokens.tolist())

    print(f"Generated Output: {output_text}")


if __name__ == "__main__":
    app()
