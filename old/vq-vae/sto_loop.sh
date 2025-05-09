#!/bin/bash

while true
do
  echo "🚀 Starting training session..."
  python training_sto.py

  if [ $? -ne 0 ]; then
    echo "❌ Python crashed. Exiting."
    exit 1
  fi

  # Find latest saved checkpoint by last modified time
  last_ckpt=$(ls -t checkpoints/ | grep ".pth" | head -n 1)
  last_epoch=$(echo "$last_ckpt" | sed -E 's/.*epoch_([0-9]+)\.pth/\1/')

  echo "✅ Last saved epoch: $last_epoch"

  if [ "$last_epoch" -ge 100 ]; then
    echo "🎉 Training complete! 100 epochs reached!"
    break
  fi

  sleep 1
done
