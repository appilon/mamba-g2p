import torch
from torch.nn.utils.rnn import pad_sequence
import sys

PAD_TOKEN = "PAD" # Consistent definition

def load_vocab(vocab_file="phoneme_vocab.txt"):
    """Loads phoneme vocabulary from a file (one symbol per line).

    Also automatically adds a PAD token mapping if not present.

    Args:
        vocab_file (str): Path to the vocabulary file.

    Returns:
        dict: phoneme_to_idx dictionary, or None if the file is not found.
    """
    phoneme_to_idx = {}
    idx_to_phoneme = [] # Keep track for consistent ordering
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symbol = line.strip()
                if symbol and symbol not in phoneme_to_idx:
                    phoneme_to_idx[symbol] = len(idx_to_phoneme)
                    idx_to_phoneme.append(symbol)

        # Ensure PAD token exists
        if PAD_TOKEN not in phoneme_to_idx:
            print(f"Adding '{PAD_TOKEN}' token to vocabulary.")
            phoneme_to_idx[PAD_TOKEN] = len(idx_to_phoneme)
            idx_to_phoneme.append(PAD_TOKEN)
        else:
            print(f"Found existing '{PAD_TOKEN}' token.")

        # Rebuild map to ensure PAD has a consistent index if it existed mid-file
        phoneme_to_idx = {symbol: idx for idx, symbol in enumerate(idx_to_phoneme)}

        print(f"Loaded phoneme vocab: {len(phoneme_to_idx)} symbols from {vocab_file}")
        return phoneme_to_idx
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_file}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading vocabulary from {vocab_file}: {e}", file=sys.stderr)
        return None


def pad_collate_fn(batch, grapheme_pad_value, phoneme_pad_idx):
    """Pads sequences in a batch for DataLoader.

    Args:
        batch (list): A list of tuples, where each tuple is
                      (grapheme_tensor, phoneme_tensor).
        grapheme_pad_value (int): The padding value for grapheme/byte sequences (e.g., 0).
        phoneme_pad_idx (int): The padding index for phoneme sequences.

    Returns:
        tuple: Contains padded grapheme sequences, padded phoneme sequences,
               original grapheme lengths, and original phoneme lengths.
               All sequences are returned as tensors.
    """
    # Separate graphemes and phonemes
    grapheme_seqs = [item[0] for item in batch]
    phoneme_seqs = [item[1] for item in batch]

    # Get original lengths (before padding)
    grapheme_lengths = torch.tensor([len(seq) for seq in grapheme_seqs], dtype=torch.long)
    phoneme_lengths = torch.tensor([len(seq) for seq in phoneme_seqs], dtype=torch.long)

    # Pad graphemes first to determine the max input length
    grapheme_padded = pad_sequence(grapheme_seqs, batch_first=True, padding_value=grapheme_pad_value)
    max_input_len = grapheme_padded.size(1) # Get the length of padded grapheme sequences

    # Pad phoneme sequences, ensuring they are also padded/truncated to max_input_len
    phoneme_padded_list = []
    for seq in phoneme_seqs:
        # Truncate if longer than max_input_len
        if len(seq) > max_input_len:
            padded_seq = seq[:max_input_len]
        # Pad if shorter
        else:
            padding_needed = max_input_len - len(seq)
            padded_seq = torch.cat([seq, torch.full((padding_needed,), phoneme_pad_idx, dtype=torch.long)])
        phoneme_padded_list.append(padded_seq)

    # Stack the padded/truncated phoneme sequences into a single tensor
    phoneme_padded = torch.stack(phoneme_padded_list, dim=0)

    # Ensure shapes match after adjustment
    assert grapheme_padded.size(1) == phoneme_padded.size(1), \
        f"Mismatch after padding: Grapheme {grapheme_padded.size(1)}, Phoneme {phoneme_padded.size(1)}"

    return grapheme_padded, phoneme_padded, grapheme_lengths, phoneme_lengths 