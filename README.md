## Model 1

### enviroment setup:
conda create -n audiocraft_env python=3.10 -y
conda activate audiocraft_env
pip install torch==2.1.0 torchvision torchaudio
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .

