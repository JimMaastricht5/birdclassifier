#!/bin/bash

PIDFILE="/home/pi/birdclassifier/run/birdclass.pid"

function cleanup() {
  rm -f "$PIDFILE"
  # Other cleanup actions
}

trap cleanup TERM INT

if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE")
  if kill -0 $PID > /dev/null 2>&1; then
    echo "Previous script is running. Terminating it gracefully..."
    kill $PID
  fi
fi

echo $$ > "$PIDFILE"

# Your script logic here
cd ~/birdclassifier
# git pull origin dev
source venv/bin/activate
python3 /home/pi/birdclassifier/birdclass.py --homedir '/home/pi/birdclassifier/' --thresholds 'USA_WI_coral.ai.inat_bird_threshold.csv' --offline True  --debug False

# clean up
cleanup
