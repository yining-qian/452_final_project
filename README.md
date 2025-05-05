## Model 1

### Environment Setup

```bash
conda create -n audiocraft_env python=3.10 -y
conda activate audiocraft_env
pip install torch==2.1.0 torchvision torchaudio
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
```

### Dataset

Download `lmd_matched.zip` from:

[Google Drive Link](https://drive.google.com/drive/folders/1C_WTDoyQkgdFOiKKjqTZpYUrehkm72Vz?dmr=1&ec=wgc-drive-globalnav-goto)

Extract and place the dataset in the project directory.

---

### Model

* Download `model1` from this repository and place it in the project directory.

Key functions implemented in the notebook:

#### **Feature Extraction**

* `extract_note_features_with_instrument(midi_file)`
  → Parses a MIDI file and returns a dataframe of note-level features (pitch, velocity, duration, instrument).

* `extract_advanced_note_features(midi_file)`
  → Extends the basic features by including `note_name`, `octave`, `start_time`, and `end_time` for richer analysis.

#### **Chord Estimation**

* `estimate_chords(note_df)`
  → Estimates chord labels from the note dataframe using a **chromagram-based algorithm**.

#### **Data Preparation**

* `SymbolicMusicDataset`
  → A PyTorch dataset wrapper that prepares tokenized sequences (pitch, velocity, duration, instrument, chord) for training.

#### **Model Architecture**

* `SymbolicMusicTransformer`
  → A Transformer-based model for symbolic music generation, conditioned on chords via **FiLM layers**.

* `ChordEncoder`
  → Encodes chord sequences into embeddings for conditioning the Transformer decoder.

#### **Music Generation**

* `save_generated_midi(pitch_seq, velocity_seq, duration_seq, instrument_seq, output_path)`
  → Converts generated token sequences back into a playable MIDI file and saves it.

---

### Customization Guide

You can:

* Add more features in `extract_advanced_note_features`
* Replace chord estimation logic in `estimate_chords`
* Modify architecture in `SymbolicMusicTransformer` or `ChordEncoder`
* Adjust output processing in `save_generated_midi`

---

## Model 2

### Environment Setup

```bash
conda create -n audiocraft_env python=3.10 -y
conda activate audiocraft_env
pip install torch==2.1.0 torchvision torchaudio
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
```

### Dataset

Download `piano.zip` from:

[Google Drive Link](https://drive.google.com/drive/folders/1C_WTDoyQkgdFOiKKjqTZpYUrehkm72Vz?dmr=1&ec=wgc-drive-globalnav-goto)

Extract and place the dataset in the project directory.

---

### Model

* Download `model2` from this repository and place it in the project directory.

Key functions implemented in the notebook:

#### **Feature Extraction**

* `extract_piano_note_features(midi_file, filename=None, composer=None)`
  → Parses a MIDI file to extract **piano-only note features**, optionally adding `filename` and `composer` metadata.

* `enrich_midi_features(df)`
  → **Enhances the note dataframe** with derived features like normalized velocity, velocity bins, or other engineered metrics.

#### **Chord Estimation**

* `estimate_chords(note_df)`
  → Estimates chord labels using a **chromagram-inspired algorithm**.

#### **Data Preparation**

* `SymbolicMusicDataset`
  → A PyTorch dataset wrapper that tokenizes sequences (pitch, velocity, duration, instrument, chord) for training.

#### **Model Architecture**

* `SymbolicMusicTransformer`
  → A Transformer-based model for symbolic music generation, conditioned on chords via **FiLM layers**.

#### **Training**

* `train_model(model, dataloader, epochs=7)`
  → Runs the training loop with customizable `epochs`, `dataloader`, and optimizer.

#### **Music Generation**

* `save_generated_midi(pitch_seq, velocity_seq, duration_seq, instrument_seq, output_path)`
  → Converts generated token sequences back into a playable MIDI file and saves it.

---

### Customization Guide

You can:

* Add more derived features in `enrich_midi_features`
* Replace chord estimation logic in `estimate_chords`
* Modify architecture in `SymbolicMusicTransformer`
* Tune training parameters in `train_model`
* Adjust output processing in `save_generated_midi`

---

## Model 3

### Environment Setup

```bash
conda create -n audiocraft_env python=3.10 -y
conda activate audiocraft_env
pip install torch==2.1.0 torchvision torchaudio
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
pip install miditok
pip install miditoolkit
```

### Dataset

Download `Maestro.zip` from:

[Google Drive Link](https://drive.google.com/drive/folders/1C_WTDoyQkgdFOiKKjqTZpYUrehkm72Vz?dmr=1&ec=wgc-drive-globalnav-goto)

Extract and place the dataset in the project directory.

---

### Model

* Download `model3` from this repository and place it in the project directory.

Key functions implemented in the notebook:

#### **Feature Extraction**

* `extract_note_features_with_instrument(midi_file)`
  → Parses a MIDI file and returns a dataframe of note-level features (pitch, velocity, duration, instrument).

* `extract_advanced_note_features(midi_file)`
  → Extends the basic features by including `note_name`, `octave`, `start_time`, and `end_time` for richer analysis.

#### **Tokenization**

* `REMI` tokenizer (from `miditok`)
  → Converts MIDI files into REMI token sequences for symbolic music modeling.

* `TokenizerConfig`
  → Defines tokenization settings (e.g., pitch range, beat resolution).

#### **Data Preparation**

* `SequenceDataset`
  → A PyTorch dataset wrapper that segments tokenized sequences into fixed-length chunks for training.

#### **Model Architecture**

* `SymbolicMusicTransformer`
  → A Transformer-based model for symbolic music generation from REMI token sequences.

#### **Training**

* `train_model(model, train_loader, num_epochs=5, lr=1e-4, device='cuda')`
  → Runs the training loop, handling optimizer, loss function, and device management.

#### **Music Generation**

* `generate_music(model, tokenizer, max_steps=512, device=device)`
  → Generates a sequence of REMI tokens from the trained model.

* `TokSequence` (from `miditok`)
  → Converts generated tokens back into a playable MIDI file.

---

### Customization Guide

You can:

* Modify tokenization rules in `TokenizerConfig`
* Change architecture in `SymbolicMusicTransformer`
* Add conditioning features (e.g., chords, instruments) into the token stream
* Tune learning rate, epochs, or optimizer in `train_model`
* Experiment with different decoding strategies in `generate_music`
