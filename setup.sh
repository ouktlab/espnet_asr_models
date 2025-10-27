#!/bin/bash
# requirement: python version over 3.10

### change these configurations according to your environment
python=python3.12

#
stage=0

# for ESPnet install
if [ $stage -le 0 ]; then
    echo "-- espnet ------- "
    sudo apt install -y ${python}-venv ${python}-dev
    sudo apt install g++ ffmpeg
    
    ${python} -m venv venv/
    . venv/bin/activate
    ${python} -m pip install espnet torchaudio transformers soxr torchcodec
    ${python} -m pip install -U espnet_model_zoo
fi

