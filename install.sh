#!/bin/sh
sudo apt update -y
sudo apt upgrade -y
DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
sudo apt install git
sudo apt install python3-pip -y
sudo apt install fluidsynth -y
pip install tensorflow fastapi uvicorn aiofiles music21 matplotlib midi2audio
git clone https://github.com/BrainBeatsUCF/Symphony.git
cd Symphony/
cd src/
uvicorn main:app --reload
