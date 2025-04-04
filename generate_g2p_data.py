import sys
import torch
from pathlib import Path
from tqdm import tqdm # Add tqdm for progress bar
import os

# --- Configuration ---
# INPUT_TEXT_FILE = "sample_text.txt" # No longer needed
DATASET_NAME = "lj_speech"           # Try this common variation
DATASET_CONFIG = None             # LJSpeech typically doesn't need a config name
DATASET_SPLIT = "train"           # Use the main train split
TEXT_FIELD = "normalized_text"  # Field likely containing cleaned text in LJSpeech
VOCAB_FILE = "phoneme_vocab.txt"
OUTPUT_DIR = "g2p_processed_data" # Directory to save processed files
SPACE_TOKEN = "_" # The symbol in our vocab to represent spaces between words from g2p
UNK_TOKEN = "UNK"   # Symbol for unknown phonemes/chars
PAD_TOKEN = "PAD"   # Symbol for padding
# -------------------

from g2p_en import G2p
from utils import load_vocab # Import from utils

def initialize_g2p():
    """Initializes the g2p-en converter, handling NLTK downloads if needed."""
    try:
        g2p = G2p()
        print("g2p-en initialized successfully.")
        return g2p
    # Removed ImportError check - Python will handle missing g2p_en package
    except LookupError as e:
        print(f"Error initializing g2p-en (NLTK data missing?): {e}")
        # Try to download NLTK data automatically

def process_text(text, g2p_instance, grapheme_vocab, phoneme_vocab):
    """Converts text to grapheme and phoneme indices."""
    try:
        phonemes = g2p_instance(text)
        # Convert graphemes/phonemes to integer indices
        grapheme_indices = [grapheme_vocab.get(g, grapheme_vocab.get('<UNK>', -1)) for g in text] # Handle potential unknown graphemes
        phoneme_indices = [phoneme_vocab.get(p, phoneme_vocab.get('<UNK>', -1)) for p in phonemes] # Handle potential unknown phonemes

        # Filter out entries with unknown tokens if necessary, or handle them
        if -1 in grapheme_indices or -1 in phoneme_indices:
            # print(f"Warning: Skipping text due to unknown tokens: {text}")
            return None # Or handle unknown tokens differently

        return torch.tensor(grapheme_indices, dtype=torch.long), torch.tensor(phoneme_indices, dtype=torch.long)
    except Exception as e:
        print(f"Error processing text: '{text}' with error: {e}")
        return None

def process_dataset(g2p, dataset_name, dataset_config, dataset_split, text_field, vocab_map, output_dir):
    """Loads dataset, converts text to phoneme indices, and saves results."""
    unk_idx = vocab_map[UNK_TOKEN]
    space_idx = vocab_map[SPACE_TOKEN]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    from datasets import load_dataset

    print(f"\n--- Loading dataset {dataset_name} (split: {dataset_split}) ---")
    try:
        # Load dataset, config (name) is often None for simpler datasets like LJSpeech
        dataset = load_dataset(dataset_name, name=dataset_config, split=dataset_split, trust_remote_code=True)
        print(f"Dataset loaded. Number of examples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        # If it failed, maybe the text field name is wrong? Try 'text'?
        print(f"Retrying with text_field='text'...")
        try:
            global TEXT_FIELD # Modify global if we retry
            TEXT_FIELD = 'text' 
            dataset = load_dataset(dataset_name, name=dataset_config, split=dataset_split, trust_remote_code=True)
            print(f"Dataset loaded with text_field='text'. Number of examples: {len(dataset)}")
        except Exception as e2:
             print(f"Error loading dataset '{dataset_name}' even with text_field='text': {e2}")
             return 0

    print(f"\n--- Processing dataset using text_field='{TEXT_FIELD}' and saving to {output_dir} ---")
    count = 0
    errors = 0
    for i, example in enumerate(tqdm(dataset, desc="Processing Examples")):
        raw_text = example.get(TEXT_FIELD) # Use the potentially updated TEXT_FIELD
        if not raw_text or not isinstance(raw_text, str):
            errors += 1
            continue
        raw_text = raw_text.strip()
        if not raw_text:
            continue
        try:
            result = process_text(raw_text, g2p, grapheme_vocab, phoneme_vocab)
            if result is None:
                errors += 1
                continue
            input_bytes, target_phonemes = result
            if input_bytes.numel() == 0 or target_phonemes.numel() == 0:
                 errors += 1
                 continue
            save_path = output_path / f"data_{count:06d}.pt"
            torch.save({
                'text': raw_text,
                'input_bytes': input_bytes,
                'target_phonemes': target_phonemes
            }, save_path)
            count += 1
        except Exception as e:
            errors += 1
            
    print(f"Finished processing. Encountered {errors} errors/skips.")
    return count

# --- Main Execution ---
if __name__ == "__main__":
    # Make sure output dir is clean before generating new data
    output_p = Path(OUTPUT_DIR)
    if output_p.exists():
        print(f"Output directory {OUTPUT_DIR} exists. Consider cleaning it before generating.")
        # Example cleaning (use with caution!):
        # import shutil
        # shutil.rmtree(OUTPUT_DIR)
        # print(f"Cleaned output directory: {OUTPUT_DIR}")
        # output_p.mkdir(parents=True, exist_ok=True)
    
    print("Loading vocabulary...")
    grapheme_vocab, phoneme_vocab = load_vocab(VOCAB_FILE)
    if grapheme_vocab is None or phoneme_vocab is None:
        print("Failed to load vocabulary. Exiting.")
        sys.exit(1)

    g2p_converter = initialize_g2p()

    processed_count = process_dataset(
        g2p_converter, 
        DATASET_NAME, 
        DATASET_CONFIG, 
        DATASET_SPLIT, 
        TEXT_FIELD, 
        grapheme_vocab, 
        OUTPUT_DIR
    )

    print(f"\n--- Finished dataset generation. Saved {processed_count} files to {OUTPUT_DIR}. ---") 