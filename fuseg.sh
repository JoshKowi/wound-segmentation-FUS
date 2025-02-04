#!/bin/bash
#SBATCH -c 16
#SBATCH --time=10:00:00
#SBATCH --job-name=train_fusegnet(MobileNet)
#SBATCH --output=train_fuseg.output
#SBATCH --mail-type=end
#SBATCH --mail-user=josua.kowalzik@mailbox.tu-dresden.de

ml restore fuseg
python train.py