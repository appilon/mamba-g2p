# Mamba-G2P-Vibe (Mambo No. 5)

This project is an experiment in "vibe coding" - exploring the implementation of a Grapheme-to-Phoneme (G2P) conversion system using the Mamba sequence model architecture. The entire development process, from initial setup to training and inference, was conducted collaboratively with the Gemini 2.5 Pro experimental model (gemini-2.5-pro-exp-03-25) within the Cursor IDE.

## Project Goal

The primary goal was to investigate the feasibility and process of building a neural G2P frontend. G2P systems are crucial components in Text-to-Speech (TTS) pipelines, converting written text (graphemes) into their phonetic representations (phonemes), which are then used by acoustic models to generate audio.

This project focuses on:
*   Using raw bytes as input to potentially handle a wider range of characters implicitly.
*   Employing the Mamba architecture, known for its efficiency in handling long sequences compared to traditional Transformers.
*   Setting up a standard PyTorch training pipeline, including data processing, model definition, training loop with validation, checkpointing, and inference.

## Process & Collaboration

This project serves as a case study in AI-assisted development. Gemini handled the bulk of the coding, including:
*   Setting up the initial project structure.
*   Writing the `G2PDataset` and `pad_collate_fn` for data loading.
*   Implementing the `NeuralG2PFrontend` model using `mamba-ssm`.
*   Developing the data generation script (`generate_g2p_data.py`) using `g2p-en` and Hugging Face `datasets` (initially targeting LibriTTS/LJSpeech).
*   Creating and refining the main training script (`train.py`) with validation, checkpointing, and loss calculation.
*   Debugging various issues related to vocabulary loading, CUDA errors, data collation, and model stability (including experiments with Mamba vs Mamba2).
*   Writing the inference script (`infer.py`) to load the trained model and predict phonemes.
*   Generating this README and the `requirements.txt`.

The human developer guided the process, set high-level goals, identified issues from outputs, and made strategic decisions (e.g., reverting from Mamba2 to Mamba for stability, deciding on training parameters). This iterative loop within Cursor allowed for rapid prototyping and exploration.

## Current Status & Results

*   **Data:** The model was trained on the LJSpeech dataset, preprocessed into byte sequences (input) and phoneme sequences (target) derived using the `g2p-en` library.
*   **Model:** A 2-layer Mamba model (`d_model=256`) was trained.
*   **Training:** Trained for 20 epochs with AdamW optimizer (LR=5e-5) and gradient clipping.
*   **Performance:** Achieved a best validation loss of ~2.99. Inference results show the model has learned a basic mapping but requires further training and potentially architectural refinements for high accuracy (see `infer.py` output for examples).

## Code Structure

*   `model.py`: Defines the `NeuralG2PFrontend` Mamba-based model.
*   `dataset.py`: Defines the `G2PDataset` for loading preprocessed data.
*   `utils.py`: Contains helper functions (`load_vocab`, `pad_collate_fn`).
*   `generate_g2p_data.py`: Script to download a dataset (e.g., LJSpeech) and convert text to byte/phoneme pairs, saving them as `.pt` files.
*   `train.py`: Main script for training the model, including validation and checkpointing.
*   `infer.py`: Script to load a trained checkpoint and perform G2P inference on sample text.
*   `phoneme_vocab.txt`: List of phoneme symbols used (derived from `g2p-en` common outputs + PAD).
*   `requirements.txt`: Python dependencies.
*   `checkpoints/`: Directory where model checkpoints are saved (created automatically).
*   `g2p_processed_data/`: Directory where processed data files are saved by `generate_g2p_data.py`.

## Setup

To set up this project on your local machine:

1. Clone the repository
   ```
   git clone https://github.com/appilon/mamba-g2p.git
   cd mamba-g2p
   ```

2. Create a virtual environment (recommended)
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```
    *Note: `mamba-ssm` requires PyTorch and CUDA development tools to be installed correctly. Follow instructions on the official `mamba-ssm` repository if you encounter installation issues.* 
    *Note: `g2p-en` may download NLTK data on first use.*

## Usage

1.  **Generate Processed Data (Optional - requires ~3GB download + processing time):**
    *   Ensure `phoneme_vocab.txt` exists or is appropriate.
    *   Clean the output directory if needed: `rm -rf g2p_processed_data/*`
    *   Run the script (configured for LJSpeech by default):
        ```bash
        python generate_g2p_data.py
        ```
2.  **Train the Model:**
    *   Ensure processed data exists in `g2p_processed_data/`.
    *   Run the training script:
        ```bash
        python train.py
        ```
    *   Checkpoints will be saved in `checkpoints/`. The best model based on validation loss will be saved as `checkpoints/best_g2p_model.pth`.
3.  **Run Inference:**
    *   Ensure a trained checkpoint exists (e.g., `checkpoints/best_g2p_model.pth`).
    *   Modify `MODEL_CHECKPOINT` in `infer.py` if using a different checkpoint.
    *   Run the inference script:
        ```bash
        python infer.py
        ```

## Future Considerations (Experimental)

*   Train for significantly more epochs.
*   Experiment with model hyperparameters (layers, dimensions, Mamba internal parameters).
*   Incorporate context awareness (e.g., processing full sentences) to handle ambiguity.
*   Compare performance with other architectures (e.g., Transformers).
*   Attempt integration with a TTS acoustic model.

## Disclaimer

This project was developed rapidly as an experiment and demonstration of AI-assisted coding. While functional, the model accuracy is preliminary and requires further work for practical application. The code reflects the iterative and sometimes trial-and-error nature of the development process guided by Gemini. 