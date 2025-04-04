# model.py

import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba # Revert back to Mamba
# from mamba_ssm import Mamba2 # Keep commented

# Simple Positional Encoding (can be refined later)
# Needed because Mamba, like Transformers, doesn't inherently know position
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Batch dim first
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input shape: [batch, seq_len, embedding_dim] """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class NeuralG2PFrontend(nn.Module):
    def __init__(self,
                 input_vocab_size=256,      # For UTF-8 bytes
                 target_vocab_size=95,     # Size of your phoneme_vocab.txt
                 d_model=256,               # Internal dimension (embedding size)
                 n_layers=6,                # Default number of Mamba layers (will be overridden by train.py)
                 d_state=64,                # Mamba state dimension (Keeping 64 for now)
                 d_conv=4,                  # Mamba convolution dimension
                 expand=2,                  # Mamba expansion factor
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.target_vocab_size = target_vocab_size

        # 1. Input Embedding
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Mamba Layers (Reverted back to Mamba)
        self.mamba_layers = nn.ModuleList([
            Mamba( # Use Mamba
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 4. Output Projection
        self.output_projection = nn.Linear(d_model, target_vocab_size)

        print(f"Initialized NeuralG2PFrontend with {n_layers} Mamba layers.") # Reverted print msg
        # You can add parameter count printing here if desired

    def forward(self, src_byte_indices):
        """
        Input: src_byte_indices (Tensor): Shape [batch_size, sequence_length]
               Contains integer indices representing input bytes (0-255).
        Output: logits (Tensor): Shape [batch_size, sequence_length, target_vocab_size]
                Raw scores for each possible phoneme at each position.
        """
        # 1. Embedding
        x = self.input_embedding(src_byte_indices) * math.sqrt(self.d_model) # Scaling often helps

        # 2. Add Positional Encoding
        x = self.pos_encoder(x)

        # 3. Pass through Mamba Layers
        for i, layer in enumerate(self.mamba_layers):
            x = layer(x)
            # Optional: Apply normalization
            # if hasattr(self, 'norms'): x = self.norms[i](x)

        x = self.norm(x)

        # 4. Project to Output Vocabulary
        logits = self.output_projection(x)

        return logits

    # NOTE: This is a simplified forward pass suitable for training where
    # the input and output sequences have the same length (like G2P).
    # For inference, especially if the output length differs significantly
    # or requires autoregressive generation (less common for G2P),
    # a separate `generate` method would be needed. 