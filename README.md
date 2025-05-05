## Model 1

### enviroment setup:
conda create -n audiocraft_env python=3.10 -y

conda activate audiocraft_env

pip install torch==2.1.0 torchvision torchaudio

git clone https://github.com/facebookresearch/audiocraft.git

cd audiocraft

pip install -e .

### dataset:
download lmd_matched.zip from

https://drive.google.com/drive/folders/1C_WTDoyQkgdFOiKKjqTZpYUrehkm72Vz?dmr=1&ec=wgc-drive-globalnav-goto

### model:
* Download `model1` from this repository and place it in the project directory.
  
Here are the key functions implemented in the notebook:

#### **Feature Extraction**

* `extract_note_features_with_instrument(midi_file)`:
  → Parses a MIDI file and returns a dataframe of note-level features (pitch, velocity, duration, instrument).

* `extract_advanced_note_features(midi_file)`:
  → Similar to the above, but also extracts `note_name`, `octave`, `start_time`, `end_time` for richer analysis.


####  **Chord Estimation**

* `estimate_chords(note_df)`:
  → Estimates chord labels from a dataframe of notes, inspired by chromagram-based chord detection.


#### **Data Preparation**

* `SymbolicMusicDataset`:
  → A PyTorch dataset wrapper that prepares tokenized sequences (pitch, velocity, duration, instrument, chord) for training.


#### **Model**

* `SymbolicMusicTransformer`:
  → Defines a Transformer-based model that generates symbolic music, with chord-conditioning applied through FiLM layers.

* `ChordEncoder`:
  → A submodule that encodes chord sequences into embeddings used to condition the Transformer decoder.


####  **Music Generation**

* `save_generated_midi(pitch_seq, velocity_seq, duration_seq, instrument_seq, output_path)`:
  → Converts generated token sequences back into a playable MIDI file and saves it.

---

#### TO Modify

You can:

* Add new features in `extract_advanced_note_features`
* Replace chord estimation logic inside `estimate_chords`
* Change model architecture in `SymbolicMusicTransformer` or `ChordEncoder`
* Adjust output processing logic inside `save_generated_midi`


## Model 2

### enviroment setup:
conda create -n audiocraft_env python=3.10 -y

conda activate audiocraft_env

pip install torch==2.1.0 torchvision torchaudio

git clone https://github.com/facebookresearch/audiocraft.git

cd audiocraft

pip install -e .

### dataset:
download piano.zip from

https://drive.google.com/drive/folders/1C_WTDoyQkgdFOiKKjqTZpYUrehkm72Vz?dmr=1&ec=wgc-drive-globalnav-goto


### Model2:

* Download `model2` from this repository and place it in the project directory.

Here are the key functions implemented in the notebook:

#### **Feature Extraction**

* `extract_piano_note_features(midi_file, filename=None, composer=None)`:
  → Parses a MIDI file and extracts **piano-only note features**, optionally tagging `filename` and `composer` metadata for later use.

* `enrich_midi_features(df)`:
  → **Enhances the extracted notes dataframe with additional features**, such as normalized velocity, discretized velocity bins, or derived metrics for downstream modeling.

#### **Chord Estimation**

* `estimate_chords(note_df)`:
  → Estimates chord labels from the note dataframe using a chromagram-inspired algorithm.

#### **Data Preparation**

* `SymbolicMusicDataset`:
  → A PyTorch dataset wrapper that prepares tokenized sequences (pitch, velocity, duration, instrument, chord) for training.

#### **Model**

* `SymbolicMusicTransformer`:
  → Defines a Transformer-based model for symbolic music generation, applying chord-conditioning via FiLM layers.

#### **Training**

* `train_model(model, dataloader, epochs=7)`:
  → Runs the training loop for the model, with optional customization for epochs, dataloader, or optimizer.

#### **Music Generation**

* `save_generated_midi(pitch_seq, velocity_seq, duration_seq, instrument_seq, output_path)`:
  → Converts generated token sequences back into a MIDI file and saves it.

---

#### TO Modify

You can:

* Add more derived features in `enrich_midi_features`
* Replace chord estimation logic inside `estimate_chords`
* Change model architecture in `SymbolicMusicTransformer`
* Adjust training settings in `train_model`
* Customize output processing logic inside `save_generated_midi`


## Model 3

### enviroment setup:
conda create -n audiocraft_env python=3.10 -y

conda activate audiocraft_env

pip install torch==2.1.0 torchvision torchaudio

git clone https://github.com/facebookresearch/audiocraft.git

cd audiocraft

pip install -e 

pip install miditok

pip install miditoolkit

### dataset:
download Maestro.zip from

https://drive.google.com/drive/folders/1C_WTDoyQkgdFOiKKjqTZpYUrehkm72Vz?dmr=1&ec=wgc-drive-globalnav-goto


### Model3:

* Download `model3` from this repository and place it in the project directory.

Here are the key functions implemented in the notebook:

#### **Feature Extraction**

* `extract_note_features_with_instrument(midi_file)`:
  → Parses a MIDI file and returns a dataframe of note-level features (pitch, velocity, duration, instrument).

* `extract_advanced_note_features(midi_file)`:
  → Similar to the above, but also extracts `note_name`, `octave`, `start_time`, `end_time` for richer analysis.


#### **Tokenization**

* `REMI` tokenizer (from `miditok`):
  → Converts MIDI files into REMI token sequences for symbolic music modeling.

* `TokenizerConfig`:
  → Defines tokenization settings (e.g., pitch range, beat resolution).

#### **Data Preparation**

* `SequenceDataset`:
  → A PyTorch dataset wrapper that segments tokenized sequences into fixed-length chunks for training.


#### **Model**

* `SymbolicMusicTransformer`:
  → Defines a Transformer-based model to generate symbolic music from REMI token sequences.


#### **Training**

* `train_model(model, train_loader, num_epochs=5, lr=1e-4, device='cuda')`:
  → Encapsulates the training loop with optimizer, loss function, and device management.


#### **Music Generation**

* `generate_music(model, tokenizer, max_steps=512, device=device)`:
  → Generates a sequence of REMI tokens from the trained model.

* `TokSequence` (from `miditok`):
  → Converts generated tokens back into a playable MIDI file.

---

#### TO Modify

You can:

* Customize tokenization rules in `TokenizerConfig`
* Change Transformer architecture in `SymbolicMusicTransformer`
* Add conditioning features (e.g., chords, instruments) into the token stream
* Adjust learning rate, epochs, or optimizer in `train_model`
* Experiment with different decoding strategies in `generate_music`

