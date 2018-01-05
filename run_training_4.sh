#!/bin/sh

#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=60000
#SBATCH -o /home/isensee/slurm_output/%x_%j.txt
#SBATCH --mail-type=ALL


python train_network.py 4
python validate_network.py 4


#Dann einfach aufrufen:
#sbatch runner.sh --job-name=MyExperimentName
