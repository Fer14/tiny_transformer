import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Initialize positional encodings
        self.register_buffer("pe", self._create_pe(max_len, d_model))

    def _create_pe(self, max_len, d_model):
        """Helper function to create positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        seq_len = x.size(1)

        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)

        # Dynamically extend positional encodings if seq_len > max_len
        if seq_len > self.max_len:
            additional_len = seq_len - self.max_len
            additional_pe = self._create_pe(additional_len, self.d_model).to(x.device)
            self.pe = torch.cat([self.pe, additional_pe], dim=1)
            self.max_len = seq_len

        # Slice positional encodings to match the input sequence length
        pe = self.pe[:, :seq_len, :].to(x.device)
        return x + pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Separate weight matrices for query (Q), key (K), and value (V)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size, q_len, d_model = query.shape
        mask = mask.to(query.device) if mask is not None else None
        _, k_len, _ = key.shape  # Key length

        # Project inputs to multi-head Q, K, V
        q = (
            self.q_proj(query)
            .reshape(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .reshape(batch_size, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .reshape(batch_size, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.softmax(scores)
        context = (attn @ v).transpose(1, 2).reshape(batch_size, q_len, d_model)

        return self.out_proj(context)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src,
        src_mask=None,
    ):
        # Self-attention + Add & Norm
        self_attn_out = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout(self_attn_out))

        # Feed-forward + Add & Norm
        ffn_out = self.ff(src)
        src = self.norm2(src + self.dropout(ffn_out))

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention + Add & Norm
        mask = self.create_tgt_mask(tgt.size(1))
        self_attn_out = self.self_attn(tgt, tgt, tgt, mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))

        # Feed-forward + Add & Norm
        ffn_out = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))

        return tgt

    def create_tgt_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=16,
        num_heads=2,
        num_layers=2,
        ff_dim=64,
        dropout=0.1,
        max_len=10,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Encoder stack (only self-attention)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Decoder stack (self-attention && cross-atention, with masking)
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # print("X", src.shape)
        # Embedding + Positional Encoding
        src = self.embedding(src)
        # print("embedding", src.shape)
        src = self.pos_encoder(src)
        # print("input", src.shape)

        # print("Y", tgt.shape)
        tgt = self.embedding(tgt)
        # print("Y-Embedding", tgt.shape)
        tgt = self.pos_encoder(tgt)
        # print("Y-input", tgt.shape)

        # Pass through encoder layers (only self-attention)
        for layer in self.encoder:
            src = layer(src, src_mask)

        # Pass through decoder layers (only self-attention)
        for layer in self.decoder:
            tgt_mask = layer.create_tgt_mask(tgt.size(1))
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return self.fc_out(tgt)

    def generate(self, input, eos_token=11, max_len=20):
        # Start with the input sequence
        output = input.clone().unsqueeze(0)  # Add batch dimension
        for _ in range(max_len):
            # Create a target mask to prevent attending to future tokens
            tgt_mask = self.decoder[0].create_tgt_mask(output.size(1)).to(output.device)

            # Forward pass through the transformer
            out = self(output, output, src_mask=None, tgt_mask=tgt_mask)

            # Get the predicted token (highest probability) from the last token in the sequence
            next_token = out[:, -1, :].argmax(
                dim=-1
            )  # Get the token with the highest probability
            next_token = next_token.unsqueeze(1)

            # Append the predicted token to the sequence
            output = torch.cat([output, next_token], dim=1)

            # If eos_token is predicted, break the loop
            if next_token.item() == eos_token:
                break

        # Remove the batch dimension before returning
        return output.squeeze(0)


def save_model(model, file_path):
    # Move the model to CPU before saving, if it is on GPU
    model.to("cpu")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_model(model, file_path, device):
    state_dict = torch.load(file_path)

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    # Move model to the specified device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model
