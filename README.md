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

---

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

