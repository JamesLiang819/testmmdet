#!/bin/bash
#SBATCH -J test
#SBATCH -t 0:0:5
#SBATCH -p tier3
#SBATCH -A puzzle
#SBATCH --mem=80G 
#SBATCH --mail-user=jcl3689@rit.edu
#SBATCH --gres=gpu:v100:4             

# spack load gcc@9.3.0/hufzekv
# spack load cuda@11.0.2/lrd2rcw
# conda activate openmmlab

# conda env list
nvidia-smi
# bash tools/dist_train.sh config/detectors/detectors_htc_r50_1x_coco.py 4 --work-dir log/fpn/r50/12e

