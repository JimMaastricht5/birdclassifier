#!/bin/bash
cd ~/birdclassifier
git pull
python3 /home/pi/birdclassifier/birdclass.py --homedir '/home/pi/birdclassifier/' --thresholds 'USA_WI_coral.ai.inat_bird_threshold.csv'

