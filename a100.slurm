#!/bin/bash
#SBATCH -J test
#SBATCH -t 24:0:0
#SBATCH -p tier3
#SBATCH -A puzzle
#SBATCH --mem=80G 
#SBATCH --mail-user=jcl3689@rit.edu
#SBATCH --gres=gpu:a100:4             

# spack load gcc@9.3.0/hufzekv
# spack load cuda@11.0.2/lrd2rcw
# conda activate openmmlab

# conda env list
nvidia-smi
# bash tools/dist_train.sh config/detectors/detectors_htc_r50_1x_coco_downsample2.py 4 --work-dir log/downsample2/r50/12e
# bash tools/dist_train.sh config/detectors/detectors_htc_r50_1x_coco.py 4 --work-dir log/test
# bash tools/dist_train.sh config/detectors/detectors_htc_r50_1x_coco.py 4 --resume-from log/fpn/r50/12e/latest.pth --work-dir log/fpn/r50/12e
bash tools/dist_test.sh config/detectors/detectors_htc_r50_1x_coco.py log/test/latest.pth 4 --work-dir log/test --eval bbox segm
