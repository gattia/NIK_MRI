## Mac install: 
mamba create --platform osx-arm64 -n dynamic_mri python=3.10 ipykernel pytorch torchvision torchaudio numpy -c pytorch-nightly
conda activate dynamic_mri
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install "wandb[media]"
pip install -e .

pip install ffmpeg-python