# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import time
import sys # Import sys for exit
import numpy as np
# from torch.cuda.amp import GradScaler, autocast # <-- Removed AMP utilities

# Import our custom modules
from model import NeuralG2PFrontend
from dataset import G2PDataset
from utils import load_vocab, pad_collate_fn
from collate import pad_collate_g2p

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "g2p_processed_data"
VOCAB_FILE = "phoneme_vocab.txt"
BATCH_SIZE = 16 # Optimal batch size found via testing
LEARNING_RATE = 5e-5 # Increased learning rate back, monitor stability
EPOCHS = 20 # Increased epochs for more substantial training
MODEL_DIMS = 256 # d_model for the NeuralG2PFrontend
NUM_LAYERS = 2 # Sticking with 2 layers (stable original Mamba)
GRAD_CLIP_VALUE = 0.5 # Keep gradient clipping
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_g2p_model.pth")
VALIDATION_SPLIT = 0.05 # Use 5% of data for validation

PAD_TOKEN = "PAD" # Must match vocab file
INPUT_VOCAB_SIZE = 256 # Explicitly set for bytes
GRAPHEME_PAD_VALUE = 0 # Use 0 for padding byte sequences

# --- Device Setup ---
print(f"Using device: {DEVICE}")

# --- Main Training Function ---
def train_model():
    print(f"Using device: {DEVICE}")

    # --- Create Checkpoint Directory ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Load Vocabulary
    print("Loading vocabulary...")
    phoneme_vocab = load_vocab(VOCAB_FILE)
    if phoneme_vocab is None:
        print("Failed to load phoneme vocabulary. Exiting.")
        return

    target_vocab_size = len(phoneme_vocab)
    try:
        phoneme_pad_idx = phoneme_vocab[PAD_TOKEN]
        print(f"Target padding index: {phoneme_pad_idx}")
    except KeyError:
        print(f"Error: PAD_TOKEN '{PAD_TOKEN}' not found in phoneme vocabulary!")
        return

    # Initialize GradScaler for mixed precision
    # scaler = GradScaler(enabled=(DEVICE.type == 'cuda')) # <-- Removed Scaler
    # print(f"GradScaler initialized (Enabled: {scaler.is_enabled()})")

    # 2. Create Dataset and DataLoader
    print("Setting up dataset and dataloader...")
    full_dataset = G2PDataset(DATA_DIR, phoneme_vocab) # Pass only phoneme vocab
    dataset_size = len(full_dataset)
    print(f"Full dataset size: {dataset_size}")

    # Split dataset
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    print(f"Splitting dataset: Train={train_size}, Validation={val_size}")
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: pad_collate_fn(b, GRAPHEME_PAD_VALUE, phoneme_pad_idx),
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, GRAPHEME_PAD_VALUE, phoneme_pad_idx),
        num_workers=4,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    print("DataLoaders ready.")

    # 3. Initialize Model
    print("Initializing model...")
    model = NeuralG2PFrontend(
        input_vocab_size=INPUT_VOCAB_SIZE, # Use 256 for bytes
        target_vocab_size=target_vocab_size, # Use loaded phoneme vocab size
        d_model=MODEL_DIMS,
        n_layers=NUM_LAYERS,
    ).to(DEVICE)
    # print(model) # Optional: print model structure

    # 4. Define Loss and Optimizer
    # CrossEntropyLoss expects logits of shape [N, C, ...] and targets [N, ...]
    # Our model outputs [Batch, SeqLen, VocabSize], targets are [Batch, SeqLen]
    # We need to reshape/permute. Ignore padding index in loss calculation.
    criterion = nn.CrossEntropyLoss(ignore_index=phoneme_pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Loss function and optimizer defined.")

    # 5. Training Loop
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    start_time_total = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.train() # Set model to training mode
        total_train_loss = 0
        num_train_batches = 0

        for i, batch in enumerate(train_loader):
            if batch is None: 
                print(f"Warning: Skipping empty batch {i}")
                continue

            # Move batch to device
            # Expecting batch format: (padded_graphemes, padded_phonemes, grapheme_lengths, phoneme_lengths)
            grapheme_indices = batch[0].to(DEVICE)
            target_phoneme_indices = batch[1].to(DEVICE)
            # target_lengths are needed if loss needs masking, but CrossEntropyLoss with ignore_index handles padding

            optimizer.zero_grad()

            # Forward pass (NO autocast)
            # with autocast(enabled=(DEVICE.type == 'cuda')):
            try:
                outputs = model(grapheme_indices)
            except Exception as e:
                print(f"\nError during model forward pass in batch {i}: {e}")
                print(f"Input shape: {grapheme_indices.shape}")
                continue

            try:
                # Reshape for CrossEntropyLoss: (batch_size * seq_len, target_vocab_size)
                outputs_flat = outputs.view(-1, target_vocab_size)
                # Reshape targets: (batch_size * seq_len)
                targets_flat = target_phoneme_indices.view(-1)

                loss = criterion(outputs_flat, targets_flat)
            except Exception as e:
                print(f"\nError during loss calculation in batch {i}: {e}")
                print(f"Logits shape: {outputs_flat.shape}, Target shape: {targets_flat.shape}")
                continue

            # Check for NaN loss BEFORE backward
            if torch.isnan(loss):
                 print(f"Warning: NaN loss detected in batch {i}, epoch {epoch}. Skipping backward/step.")
                 continue 

            # Backward pass (NO scaler)
            try:
                # scaler.scale(loss).backward()
                loss.backward()
                # Optional: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_VALUE)
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
            except Exception as e:
                 print(f"\nError during backward/optimizer step in batch {i}: {e}")
                 continue

            total_train_loss += loss.item() 
            num_train_batches += 1

            # Print progress periodically
            if (i + 1) % 100 == 0: 
                 print(f"Epoch {epoch}/{EPOCHS} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # End of epoch
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        epoch_duration = time.time() - epoch_start_time

        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad(): # Disable gradient calculations for validation
            for i, batch in enumerate(val_loader):
                grapheme_indices = batch[0].to(DEVICE)
                target_phoneme_indices = batch[1].to(DEVICE)

                try:
                    outputs = model(grapheme_indices)
                    outputs_flat = outputs.view(-1, target_vocab_size)
                    targets_flat = target_phoneme_indices.view(-1)
                    loss = criterion(outputs_flat, targets_flat)

                    if not torch.isnan(loss):
                         total_val_loss += loss.item()
                         num_val_batches += 1

                except Exception as e:
                    print(f"Error during validation forward pass in batch {i}: {e}")
                    continue

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

        print("-" * 50)
        print(f"End of Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Duration: {epoch_duration:.2f}s")
        print("-" * 50)

        # --- Save Checkpoint ---
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"g2p_model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** New best model saved based on validation loss: {best_val_loss:.4f} ***")

    total_duration = time.time() - start_time_total
    print("\n--- Training Finished ---")
    print(f"Total Training Duration: {total_duration:.2f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model state dictionary saved to: {BEST_MODEL_PATH}")


# --- Run Training ---
if __name__ == "__main__":
    train_model()