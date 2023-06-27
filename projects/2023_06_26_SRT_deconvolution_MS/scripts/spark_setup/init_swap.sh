#!/bin/bash

echo "Initializing swap with $1 with $2 100Mb units"

sudo swapoff -a
sudo dd if=/dev/zero of=$1 bs=100M count=$2
sudo chmod 0600 $1
sudo mkswap $1  # Set up a Linux swap area
sudo swapon $1  # Turn the swap on
