#!/bin/bash
cd ~/birdclassifier
git pull origin dev
source venv/bin/activate
python3 /home/pi/birdclassifier/cameratest2.py