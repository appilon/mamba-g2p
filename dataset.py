# dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import warnings

class G2PDataset(Dataset):
    """Custom PyTorch Dataset for loading preprocessed G2P data."""
    def __init__(self, data_dir, phoneme_vocab):
        """Initializes the dataset.

        Args:
            data_dir (str): The directory containing the processed .pt files.
            phoneme_vocab (dict): Dictionary mapping phonemes to indices.
        """
        self.data_dir = Path(data_dir)
        self.phoneme_vocab = phoneme_vocab # Store phoneme vocab
        self.file_list = sorted([p for p in self.data_dir.glob("*.pt") if p.is_file()])

        if not self.file_list:
            raise FileNotFoundError(f"No .pt files found in directory: {data_dir}")

        print(f"Found {len(self.file_list)} processed data files in {data_dir}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Loads and returns a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (grapheme_indices, phoneme_indices) tensors for the sample.
                   Returns None if there's an error loading the file.
        """
        file_path = self.file_list[idx]
        try:
            # Suppress the specific FutureWarning about weights_only
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
                data_item = torch.load(file_path)

            # Assuming the .pt file contains a dictionary with keys
            # 'input_bytes' and 'target_phonemes' storing the tensors.
            grapheme_indices = data_item['input_bytes']
            phoneme_indices = data_item['target_phonemes']

            # Basic validation (optional but recommended)
            if not isinstance(grapheme_indices, torch.Tensor) or not isinstance(phoneme_indices, torch.Tensor):
                 print(f"Warning: Invalid data format in {file_path}. Skipping.")
                 return self.__getitem__((idx + 1) % len(self)) # Skip and try next
            if grapheme_indices.numel() == 0 or phoneme_indices.numel() == 0:
                 print(f"Warning: Empty tensor found in {file_path}. Skipping.")
                 return self.__getitem__((idx + 1) % len(self)) # Skip and try next

            return grapheme_indices, phoneme_indices
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return self.__getitem__((idx + 1) % len(self)) # Skip and try next
        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}")
            return self.__getitem__((idx + 1) % len(self)) # Skip and try next

# --- Example Usage (for testing the Dataset class itself) ---
if __name__ == '__main__':
    print("Testing G2PDataset...")
    DATA_DIR = "g2p_processed_data"

    try:
        dataset = G2PDataset(data_dir=DATA_DIR)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            print("\nLoading first sample:")
            first_sample = dataset[0]
            if first_sample:
                input_data, target_data = first_sample
                print(f" Input shape: {input_data.shape}, dtype: {input_data.dtype}")
                print(f" Input data (first 20): {input_data[:20].tolist()}")
                print(f" Target shape: {target_data.shape}, dtype: {target_data.dtype}")
                print(f" Target data (first 20): {target_data[:20].tolist()}")
            else:
                print(" Failed to load the first sample.")

        if len(dataset) > 2:
             print("\nLoading third sample:")
             third_sample = dataset[2]
             if third_sample:
                  input_data, target_data = third_sample
                  print(f" Input shape: {input_data.shape}")
                  print(f" Target shape: {target_data.shape}")
             else:
                  print(" Failed to load the third sample.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 