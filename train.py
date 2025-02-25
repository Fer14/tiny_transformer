import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BracketDataset
from model import TinyTransformer, save_model

torch.cuda.empty_cache()


def train_model(model, dataloader, epochs=5, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_batches = len(dataloader) * epochs

    with tqdm(total=total_batches, desc="Training") as pbar:
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (input_tokens, output_tokens) in enumerate(dataloader):
                input_tokens, output_tokens = (
                    input_tokens.to(device),
                    output_tokens.to(device),
                )

                optimizer.zero_grad()

                tgt_input = output_tokens[:, :-1]  # Remove <EOS> token
                tgt_output = output_tokens[:, 1:]  # Remove <SOS> token

                # print(
                #     "input_tokens: ",
                #     input_tokens.shape,
                #     input_tokens,
                #     dataloader.dataset.decode(input_tokens[0]),
                # )
                # print(
                #     "tgt_input: ",
                #     tgt_input.shape,
                #     tgt_input,
                #     dataloader.dataset.decode(tgt_input[0]),
                # )
                # print(
                #     "tgt_output: ",
                #     tgt_output.shape,
                #     tgt_output,
                #     dataloader.dataset.decode(tgt_output[0]),
                # )
                # exit()

                predictions = model(input_tokens, tgt_input)

                loss = loss_fn(
                    predictions.reshape(-1, predictions.size(-1)),
                    tgt_output.reshape(-1),
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_description_str(f"Training Epoch {epoch + 1}/{epochs}")
                pbar.update(1)

            avg_loss = total_loss / len(dataloader)  # Compute average loss per epoch
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    return model


def main():
    dataset = BracketDataset(num_samples=1000000, max_len=10)
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
    )

    model = TinyTransformer(
        vocab_size=len(dataset.VOCAB),
        d_model=16,
        num_heads=4,
        num_layers=4,
        ff_dim=128,
        max_len=dataset.max_len,
    )

    try:
        model = train_model(model, dataloader, epochs=100)
    except Exception as e:
        print(e)
    finally:
        save_model(model, "model.pth")

        with open("vocab.json", "w") as f:
            json.dump(dataset.VOCAB, f)

        print(dataset.VOCAB)


if __name__ == "__main__":
    main()
