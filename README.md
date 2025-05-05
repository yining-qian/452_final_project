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
* supports **chord conditioning** (input chord sequences influence generated music)
* generates **multi-instrument symbolic music** (outputs pitch, velocity, duration, instrument)
* uses **relative multihead attention** and **FiLM conditioning layers**


You can modify or extend the model:

* **change chord encoder**: update the encoder layers in `ChordEncoder` to experiment with different architectures
* **add more conditioning**: integrate tempo, key signature, or style embeddings into the decoder
* **swap output head**: adjust `Linear` layer for different vocab sizes or additional prediction targets
