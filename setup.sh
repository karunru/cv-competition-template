#!/bin/bash

apt install curl vim git gcc htop

mkdir -p /content/petfinder-pawpularity-score/

curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

echo 'eval "$(pyenv init --path)"' >>~/.bashrc

echo 'eval "$(pyenv init -)"' >> ~/.bashrc

echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

source ~/.profile
source ~/.bashrc

env PYTHONPATH=
pyenv install miniconda3-4.7.12
pyenv global miniconda3-4.7.12

conda update --channel defaults --all --yes

conda create -n rapids-21.12 -c rapidsai -c nvidia -c conda-forge \
    cudf=21.12 cuml=21.12 python=3.8 cudatoolkit=11.2 --yes

conda activate rapids-21.12

echo 'pyenv global miniconda3-4.7.12' >> ~/.bashrc
echo 'conda activate rapids-21.12' >> ~/.bashrc

mkdir ~/.kaggle
cp /content/drive/MyDrive/Kaggle/kaggle.json ~/.kaggle/
cp /content/drive/MyDrive/Kaggle/tmux.conf ~/.tmux.conf
tmux source ~/.tmux.conf

cd /content/petfinder-pawpularity-score/dataset
/root/.pyenv/versions/miniconda3-4.7.12/envs/rapids-21.12/bin/kaggle competitions download -c petfinder-pawpularity-score
unzip -q petfinder-pawpularity-score.zip
/root/.pyenv/versions/miniconda3-4.7.12/envs/rapids-21.12/bin/kaggle datasets download -d karunru/petfinder1-2-yolox-x-cropped-dataset
unzip -q petfinder1-2-yolox-x-cropped-dataset.zip
mv pet2/yolox_x_crop .; rm -rf pet2
mkdir -p petfinder-adoption-prediction/yolox_x_crop/
mv pet1/yolox_x_train/ petfinder-adoption-prediction/yolox_x_crop/train_images/
mv pet1/yolox_x_test/ petfinder-adoption-prediction/yolox_x_crop/test_images/
rm -rf pet1
