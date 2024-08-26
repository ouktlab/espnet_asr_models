#!/bin/bash
# requirement: python version over 3.10

### change these configurations according to your environment
python=python3.10

#
stage=0

# for ESPnet install
if [ $stage -le 0 ]; then
    echo "-- espnet ------- "
    sudo apt install -y ${python}-venv ${python}-dev
    
    ${python} -m venv venv/
    . venv/bin/activate
    ${python} -m pip install espnet torchaudio
    ${python} -m pip install -U espnet_model_zoo
fi

