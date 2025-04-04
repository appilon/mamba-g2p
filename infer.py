import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Assuming model.py and utils.py are accessible
from model import NeuralG2PFrontend
from utils import load_vocab # Correct import from utils

# --- Configuration --- (Adjust as needed) ---
MODEL_CHECKPOINT = "checkpoints/best_g2p_model.pth" # Path to the best saved model
VOCAB_FILE = "phoneme_vocab.txt"
INPUT_VOCAB_SIZE = 256 # Bytes
MODEL_DIMS = 256
NUM_LAYERS = 2
PAD_TOKEN = "PAD" # Ensure this matches vocab and training
GRAPHEME_PAD_VALUE = 0 # Padding value for input bytes
# Max sequence length for padding during inference (adjust if needed)
MAX_INFER_LEN_INPUT = 200

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Helper Functions ---
def text_to_bytes(text):
    """Converts text to a list of bytes."""
    return list(text.encode('utf-8'))

def preprocess_input(text, max_len, pad_value):
    """Converts text to padded byte tensor."""
    byte_sequence = text_to_bytes(text)
    seq_len = len(byte_sequence)

    if seq_len > max_len:
        print(f"Warning: Input text '{text}' (length {seq_len}) exceeds max_len {max_len}. Truncating.")
        byte_sequence = byte_sequence[:max_len]
        seq_len = max_len

    # Pad sequence with the specified pad_value
    padded_sequence = byte_sequence + [pad_value] * (max_len - seq_len)
    input_tensor = torch.tensor(padded_sequence, dtype=torch.long).unsqueeze(0) # Add batch dim
    return input_tensor.to(DEVICE), seq_len # Also return original length

def decode_phonemes(model, input_tensor, original_input_len, phoneme_pad_idx, idx_to_phoneme):
    """Generates phoneme indices using a simple forward pass and argmax."""
    model.eval()
    predicted_indices = []

    with torch.no_grad():
        logits = model(input_tensor) # Shape: [1, SeqLen (padded), VocabSize]

        # Get the most likely phoneme index for each position *up to the original input length*
        # Note: Output seq length matches padded input length due to training setup
        output_indices = torch.argmax(logits, dim=-1).squeeze(0) # Shape: [SeqLen (padded)]

        # Consider only the output corresponding to the original input length
        # (Model might output PADs for padded inputs, which we can ignore)
        # We take original_input_len as a heuristic, adjust if needed
        output_list = output_indices[:original_input_len].cpu().tolist()

        for idx in output_list:
            if idx == phoneme_pad_idx:
                # Optional: Stop if PAD is predicted within the original length region
                # break
                pass # More likely, just skip printing PADs
            else:
                predicted_indices.append(idx)

    # Convert indices to phonemes
    predicted_phonemes = [idx_to_phoneme.get(idx, f"?{idx}?") for idx in predicted_indices]
    return predicted_phonemes


# --- Main Inference Logic ---
def main():
    # 1. Load Vocabulary
    print(f"Loading phoneme vocabulary from {VOCAB_FILE}...")
    phoneme_vocab = load_vocab(VOCAB_FILE)
    if phoneme_vocab is None:
        print("Failed to load vocabulary. Exiting.")
        sys.exit(1)

    target_vocab_size = len(phoneme_vocab)
    try:
        phoneme_pad_idx = phoneme_vocab[PAD_TOKEN]
        print(f"Phoneme vocabulary loaded ({target_vocab_size} symbols). PAD index: {phoneme_pad_idx}")
    except KeyError:
        print(f"Error: PAD token '{PAD_TOKEN}' not found in phoneme vocabulary!")
        sys.exit(1)

    # Create reverse mapping (index to phoneme)
    idx_to_phoneme = {v: k for k, v in phoneme_vocab.items()}

    # 2. Initialize Model
    print("Initializing model...")
    try:
        model = NeuralG2PFrontend(
            input_vocab_size=INPUT_VOCAB_SIZE, # 256 for bytes
            target_vocab_size=target_vocab_size,
            d_model=MODEL_DIMS,
            n_layers=NUM_LAYERS
        ).to(DEVICE)
        print(f"Initialized model with {NUM_LAYERS} layers.")
    except Exception as e:
        print(f"Error initializing model structure: {e}")
        sys.exit(1)

    # 3. Load Model Weights
    print(f"Loading weights from {MODEL_CHECKPOINT}...")
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Checkpoint file not found at {MODEL_CHECKPOINT}")
        sys.exit(1)
    try:
        # Load the entire checkpoint dictionary
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)

        # Extract the model state dictionary
        model_state_dict = checkpoint['model_state_dict']

        # Handle potential DataParallel wrapping (unlikely here, but good practice)
        if isinstance(model, torch.nn.DataParallel):
             model.module.load_state_dict(model_state_dict)
        else:
             model.load_state_dict(model_state_dict)

        print(f"Model weights loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")
    except KeyError:
         print("Error: Checkpoint dictionary missing 'model_state_dict'. Loading raw state dict...")
         # Fallback for checkpoints that only contain the state dict itself
         try:
             model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
             print("Model weights loaded successfully (raw state dict).")
         except Exception as e_raw:
             print(f"Error loading raw state dict: {e_raw}")
             sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading model weights (likely architecture mismatch): {e}")
        print("Ensure MODEL_DIMS and NUM_LAYERS match the saved checkpoint.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    model.eval() # Set model to evaluation mode

    # 4. Example Inference
    print("\n--- Running Inference Examples ---")
    example_texts = [
        "hello",
        "world",
        "testing",
        "phonetics",
        "sequence",
        "strength",
        "neural",
        "mamba",
        "This is a test sentence.",
        "How well does this grapheme to phoneme model work?",
        "Hexgrad", # Example OOV word
        "pronunciation",
        "ambiguity",
    ]

    for text in example_texts:
        print(f"\nInput Text: '{text}'")
        try:
            # Preprocess
            input_tensor, original_len = preprocess_input(text, MAX_INFER_LEN_INPUT, GRAPHEME_PAD_VALUE)

            # Predict
            predicted_phonemes = decode_phonemes(model, input_tensor, original_len, phoneme_pad_idx, idx_to_phoneme)
            phoneme_string = " ".join(predicted_phonemes)

            print(f"Predicted Phonemes: {phoneme_string}")

        except Exception as e:
            print(f"Error during inference for '{text}': {e}")

# --- Run Inference ---
if __name__ == "__main__":
    main() 