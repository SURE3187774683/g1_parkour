#! /bin/bash
DEVICE=cuda:1

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/train.py \
--task g1_distill \
--rl_device $DEVICE \
--sim_device $DEVICE \
--headless \
# --run_name V1_ \
# --run_name From_Apr18_15-08-13_TerrainPerlin_50000 \
# --load_run=Apr15_14-51-52_Tron1_10skills_fromApr11_17-14-24 \
# --checkpoint=74300 \