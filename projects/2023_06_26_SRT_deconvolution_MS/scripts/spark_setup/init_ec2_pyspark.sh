#!/bin/bash

echo "Initializing disk and swap for 4xlarge node..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

sudo mkdir /temp

# Create one partition combining two SSDs. Mount on /temp
sudo sh ${SCRIPT_DIR}/combine_two_ssd.sh nvme2n1 nvme3n1 /temp

# Create 80Gb swap space
sudo sh ${SCRIPT_DIR}/init_swap.sh /temp/swapfile 800
