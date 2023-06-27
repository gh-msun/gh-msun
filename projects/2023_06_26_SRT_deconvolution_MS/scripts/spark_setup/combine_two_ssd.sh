#!/bin/bash

echo "Combining /dev/$1 and /dev/$2. Mounting on $3."

sudo pvcreate /dev/$1  /dev/$2
sudo vgcreate LVMVolGroup /dev/$1  /dev/$2
sudo lvcreate -l 100%FREE -n ssd LVMVolGroup

#sudo file -s /dev/$1
sudo mkfs -t xfs /dev/LVMVolGroup/ssd
sudo mount /dev/LVMVolGroup/ssd $3
sudo chown -R ubuntu:ubuntu $3
